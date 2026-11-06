#include "robottrace/events.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <iostream>
#include <memory>

using namespace robottrace;
using Message = std_msgs::msg::UInt64MultiArray;
using Trigger = std_srvs::srv::Trigger;

struct Options {
    std::string stage, run, dir, base, source = "source:1";
    uint64_t run_hi, run_lo, count, work_ns;
    size_t depth, capacity;
    double rate;

    Options(int argc, char **argv) {
        if (argc != 10)
            throw std::runtime_error("Usage: pipeline STAGE UUIDHEX OUTPUT COUNT RATE "
                                     "DEPTH WORK_MS CAPACITY");
        stage = argv[1];
        run = argv[2];
        dir = argv[3];
        count = std::stoull(argv[4]);
        rate = std::stod(argv[5]);
        depth = std::stoull(argv[6]);
        double work = std::stod(argv[7]);
        capacity = std::stoull(argv[8]);
        // 预留的来源标识在消息载荷中显式携带，不从话题名称推断。
        if (std::string(argv[9]) != "1")
            throw std::runtime_error("R1-01 supports source token 1 only");
        if (run.size() != 32 ||
            run.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("Invalid run UUID");
        if (stage != "source" && stage != "processor" && stage != "consumer")
            throw std::runtime_error("Invalid stage");
        if (!count || count > 10000 || !(rate > 0 && rate <= 1000) || !depth ||
            depth > 10000 || work < 0 || work > 1000 || capacity > 1000000)
            throw std::runtime_error("Invalid workload bounds");
        work_ns = uint64_t(work * 1000000);
        run_hi = std::stoull(run.substr(0, 16), nullptr, 16);
        run_lo = std::stoull(run.substr(16), nullptr, 16);
        base = "/robottrace/r_" + run;
    }

    bool valid(const Message &m) const {
        return m.data.size() == 6 && m.data[0] == 1 && m.data[1] == run_hi &&
            m.data[2] == run_lo && m.data[3] == 1 && m.data[4] >= 1 &&
            m.data[4] <= count;
    }
};

int main(int argc, char **argv) {
    try {
        Options o(argc, argv);
        Clock clock;
        Buffer events(o.capacity);
        rclcpp::init(argc, argv);
        auto node = std::make_shared<rclcpp::Node>("robottrace_" + o.stage);
        auto qos =
            rclcpp::QoS(rclcpp::KeepLast(o.depth)).reliable().durability_volatile();
        rclcpp::Publisher<Message>::SharedPtr pub;
        rclcpp::Subscription<Message>::SharedPtr sub;
        rclcpp::Service<Trigger>::SharedPtr ready_service;
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        const auto started = now();
        uint64_t invalid = 0, matched_at = 0, published = 0, publish_errors = 0;
        bool safety = false;
        std::vector<unsigned char> seen(o.count + 1, 0);
        uint64_t unique_received = 0;
        if (o.stage == "source" || o.stage == "processor")
            pub = node->create_publisher<Message>(
                o.base + (o.stage == "source" ? "/input" : "/output"), qos);
        if (o.stage != "source") {
            sub = node->create_subscription<Message>(
                o.base + (o.stage == "processor" ? "/input" : "/output"), qos,
                [&](Message::ConstSharedPtr message) {
                    const auto entry = now();
                    if (!o.valid(*message) || message->data[5] > entry) {
                        ++invalid;
                        return;
                    }
                    const auto seq = message->data[4];
                    events.record_at(seq, Kind::Entry, entry);
                    if (seen[seq])
                        ++invalid;
                    else {
                        seen[seq] = 1;
                        ++unique_received;
                    }
                    if (o.stage == "processor") {
                        sleep_until(now() + o.work_ns);
                        events.record(seq, Kind::PublishBegin);
                        try {
                            pub->publish(*message);
                            events.record(seq, Kind::PublishEnd);
                            ++published;
                        } catch (const std::exception &) {
                            ++publish_errors;
                        } // 即使关闭过程打断发布，也保留已有观测，
                          // 供后续刷新到文件。
                    }
                    events.record(seq, Kind::Exit);
                });
        }
        if (o.stage == "processor")
            ready_service = node->create_service<Trigger>(o.base + "/ready",
                [&](const std::shared_ptr<Trigger::Request>,
                    std::shared_ptr<Trigger::Response> response) {
                    response->success = pub->get_subscription_count() == 1 &&
                        sub->get_publisher_count() == 1;
                    response->message = response->success
                        ? "Both processor DDS endpoints matched"
                        : "Waiting for DDS endpoints";
                });
        {
            auto f = output(o.dir + "/" + o.stage + ".started");
            f << started << '\n';
        }
        if (o.stage == "source") {
            auto client = node->create_client<Trigger>(o.base + "/ready");
            const auto deadline = now() + 25 * second;
            bool matched = false;
            while (rclcpp::ok() && now() < deadline) {
                if (!client->wait_for_service(std::chrono::milliseconds(100)))
                    continue;
                auto future =
                    client->async_send_request(std::make_shared<Trigger::Request>());
                auto result = executor.spin_until_future_complete(
                    future, std::chrono::milliseconds(500));
                if (result == rclcpp::FutureReturnCode::SUCCESS &&
                    future.get()->success && pub->get_subscription_count() == 1) {
                    matched = true;
                    break;
                }
                if (result == rclcpp::FutureReturnCode::TIMEOUT)
                    client->remove_pending_request(future);
            }
            if (!matched)
                throw std::runtime_error(
                    "Pipeline matching/readiness deadline exceeded");
            matched_at = now();
            {
                auto f = output(o.dir + "/matched.json");
                f << "{\"timestamp_ns\":" << matched_at
                  << ",\"source_matched_subscribers\":" << pub->get_subscription_count()
                  << ",\"processor_ready_response\":true}\n";
            }
            Message message;
            message.data = {1, o.run_hi, o.run_lo, 1, 0, 0};
            const auto first = now();
            const auto period = uint64_t(double(second) / o.rate);
            for (uint64_t seq = 1; seq <= o.count && rclcpp::ok(); ++seq) {
                if (now() > first + 20 * second) {
                    safety = true;
                    break;
                }
                sleep_until(first + (seq - 1) * period);
                message.data[4] = seq;
                const auto generated = now();
                message.data[5] = generated;
                events.record_at(seq, Kind::Generate, generated);
                events.record(seq, Kind::PublishBegin);
                try {
                    pub->publish(message);
                    events.record(seq, Kind::PublishEnd);
                    ++published;
                } catch (const std::exception &) {
                    ++publish_errors;
                    break;
                }
            }
            {
                auto f = output(o.dir + "/published.json");
                f << "{\"completed_ns\":" << now() << ",\"published\":" << published
                  << "}\n";
            }
        }
        while (rclcpp::ok() && now() < started + 60 * second)
            executor.spin_once(std::chrono::milliseconds(10));
        if (rclcpp::ok())
            safety = true;
        events.flush(o.dir + "/" + o.stage + ".jsonl", o.run, o.source, o.stage, clock);
        auto f = output(o.dir + "/" + o.stage + ".status.json");
        f << "{\"stage\":" << quote(o.stage) << ",\"process_id\":" << getpid()
          << ",\"stored\":" << events.size() << ",\"dropped\":" << events.dropped()
          << ",\"flushed\":true,\"clock_domain\":" << quote(clock.domain)
          << ",\"boot_id\":" << quote(clock.boot_id)
          << ",\"time_offsets\":" << quote(clock.offsets)
          << ",\"started_ns\":" << started << ",\"ended_ns\":" << now()
          << ",\"matched_at_ns\":" << matched_at << ",\"published\":" << published
          << ",\"publish_errors\":" << publish_errors
          << ",\"unique_received\":" << unique_received
          << ",\"invalid_messages\":" << invalid
          << ",\"safety_deadline\":" << (safety ? "true" : "false") << "}\n";
        f.close();
        if (rclcpp::ok())
            rclcpp::shutdown();
        return safety || invalid || publish_errors ? 2 : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        if (rclcpp::ok())
            rclcpp::shutdown();
        return 1;
    }
}
