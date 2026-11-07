#include "inference.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>

using namespace robottrace;
using namespace inference;

namespace {
    void atomic_json(const std::string &path, const std::string &text) {
        const auto temporary = path + ".tmp";
        {
            auto file = output(temporary);
            file << text << '\n';
            file.close();
        }
        if (std::rename(temporary.c_str(), path.c_str()))
            throw std::runtime_error("Cannot publish control file: " + path);
    }

    std::string qos_json(const rclcpp::QoS &qos) {
        const auto &v = qos.get_rmw_qos_profile();
        std::ostringstream out;
        out << "{\"depth\":" << v.depth << ",\"reliability\":"
            << quote(v.reliability == RMW_QOS_POLICY_RELIABILITY_RELIABLE ? "RELIABLE"
                                                                          : "OTHER")
            << ",\"durability\":"
            << quote(v.durability == RMW_QOS_POLICY_DURABILITY_VOLATILE ? "VOLATILE"
                                                                        : "OTHER")
            << ",\"history\":"
            << quote(v.history == RMW_QOS_POLICY_HISTORY_KEEP_LAST ? "KEEP_LAST"
                                                                   : "OTHER")
            << '}';
        return out.str();
    }

    struct Record {
        uint64_t sequence = 0, generated = 0, entry = 0;
        uint32_t sample = 0;
        Call call;
        bool valid = false;
    };

    void flush_records(const std::string &path, const std::vector<Record> &records,
        const Identity &identity, const Clock &clock, bool processor) {
        auto file = output(path);
        file << std::setprecision(9);
        for (const auto &r : records) {
            file << "{\"run_id\":" << quote(identity.run)
                 << ",\"source_id\":\"source:1\",\"sequence\":" << r.sequence
                 << ",\"sample_id\":" << r.sample << ",\"generated_ns\":" << r.generated
                 << ",\"entry_ns\":" << r.entry
                 << ",\"clock_domain\":" << quote(clock.domain)
                 << ",\"valid\":" << (r.valid ? "true" : "false");
            if (processor)
                file << ",\"input_payload_verified\":true,\"api_begin_ns\":"
                     << r.call.begin << ",\"api_end_ns\":" << r.call.end
                     << ",\"api_returned\":" << (r.call.returned ? "true" : "false");
            file << ",\"logits\":";
            if (!r.call.output_available)
                file << "null";
            else {
                file << '[';
                for (size_t i = 0; i < r.call.logits.size(); ++i) {
                    if (i)
                        file << ',';
                    if (std::isfinite(r.call.logits[i]))
                        file << r.call.logits[i];
                    else
                        file << "null";
                }
                file << ']';
            }
            file << "}\n";
        }
        file.close();
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 7 && argc != 8 && argc != 10)
            throw std::runtime_error("Usage: inference_pipeline ROLE RUN OUTPUT COUNT "
                                     "FIXTURES MODEL [RATE [--intra-op-threads 1|4]]");
        const std::string role = argv[1], dir = argv[3], fixtures_path = argv[5],
                          model_path = argv[6];
        if (role != "source" && role != "processor" && role != "consumer")
            throw std::runtime_error("Invalid role");
        int requested_threads = 1;
        if (argc == 10) {
            if (role != "processor" || std::string(argv[8]) != "--intra-op-threads")
                throw std::invalid_argument("Only Processor accepts --intra-op-threads");
            requested_threads = processor_threads(argv[9]);
        }
        const Identity identity(argv[2], std::stoull(argv[4]));
        const Schedule schedule(identity.count, argc >= 8 ? std::stod(argv[7]) : 5.0);
        const Clock clock;
        Buffer events(4096);
        const auto started = now();
        std::string error;
        std::string inference_configuration = "null";
        uint64_t received = 0, invalid = 0, duplicates = 0, published = 0,
                 operations = 0;
        uint64_t publish_attempts = 0, observed_unique = 0, schedule_start = 0;
        uint64_t valid_count = 0, warmup_begin = 0, warmup_end = 0, warmups = 0,
                 sessions = 0, diagnostic_dropped = 0;
        uint64_t model_loaded = 0, ros_initialized = 0, node_created = 0,
                 input_created = 0, output_created = 0;
        std::vector<Record> records;
        records.reserve(identity.count);
        try {
            Fixtures fixtures(fixtures_path);
            std::unique_ptr<CpuModel> model;
            if (role == "processor") {
                model = std::make_unique<CpuModel>(model_path, requested_threads);
                model_loaded = now();
                sessions = 1;
                inference_configuration =
                    "{\"requested_intra_op_threads\":" + std::to_string(requested_threads) +
                    ",\"intra_op_threads\":" + std::to_string(model->intra_op_num_threads()) +
                    ",\"inter_op_threads\":1,\"execution_mode\":\"sequential\","
                    "\"graph_optimization\":\"basic\",\"cpu_arena\":true,"
                    "\"provider\":\"CPUExecutionProvider\",\"onnxruntime\":\"1.20.1\","
                    "\"applied_via\":\"ORT setter and Session construction succeeded\"}";
                warmup_begin = now();
                for (uint32_t warmup = 0; warmup < 5; ++warmup) {
                    const auto sample = warmup % sample_count;
                    Call call;
                    model->run(fixtures.input[sample], call);
                    if (!fixtures.correct(call.logits, sample))
                        throw std::runtime_error("Warm-up reference mismatch");
                    ++warmups;
                }
                warmup_end = now();
            }
            rclcpp::init(argc, argv);
            ros_initialized = now();
            auto node = std::make_shared<rclcpp::Node>("robottrace_inference_" + role);
            node_created = now();
            rclcpp::executors::SingleThreadedExecutor executor;
            executor.add_node(node);
            auto fail = [&](const std::string &reason) {
                if (error.empty())
                    error = reason;
            };
            const auto qos =
                rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
            const std::string topic = "/robottrace/inference_" + identity.run;
            rclcpp::Publisher<Input>::SharedPtr source_pub;
            rclcpp::Publisher<Result>::SharedPtr result_pub;
            rclcpp::Subscription<Input>::SharedPtr input_sub;
            rclcpp::Subscription<Result>::SharedPtr result_sub;
            Results results;
            std::array<bool, max_messages + 1> processor_seen{};
            auto retain = [&](const Record &record) {
                if (records.size() < identity.count)
                    records.push_back(record);
                else {
                    ++diagnostic_dropped;
                    fail("Diagnostic capacity exceeded");
                }
            };
            if (role == "source") {
                source_pub = node->create_publisher<Input>(topic + "/input", qos);
                output_created = now();
            }
            if (role == "processor") {
                result_pub = node->create_publisher<Result>(topic + "/output", qos);
                output_created = now();
                input_sub = node->create_subscription<Input>(
                    topic + "/input", qos, [&](Input::ConstSharedPtr message) {
                        const auto entry = now();
                        ++received;
                        if (!identity.valid(*message, entry) ||
                            !fixtures.payload(*message)) {
                            ++invalid;
                            fail("Invalid input identity/sample/tensor");
                            return;
                        }
                        const auto seq = message->sequence;
                        events.record_at(seq, Kind::Entry, entry);
                        if (processor_seen[seq]) {
                            ++duplicates;
                            fail("Duplicate input");
                            events.record(seq, Kind::Exit);
                            return;
                        }
                        processor_seen[seq] = true;
                        ++observed_unique;
                        Record record;
                        record.sequence = seq;
                        record.generated = message->generated_ns;
                        record.sample = message->sample_id;
                        record.entry = entry;
                        try {
                            Tensor input = message->input;
                            model->run(input, record.call);
                            if (!fixtures.correct(
                                    record.call.logits, message->sample_id))
                                throw std::runtime_error(
                                    "Processor reference mismatch");
                            Result result;
                            result.run_hi = message->run_hi;
                            result.run_lo = message->run_lo;
                            result.source_id = message->source_id;
                            result.sequence = seq;
                            result.generated_ns = message->generated_ns;
                            result.sample_id = message->sample_id;
                            result.logits = record.call.logits;
                            ++publish_attempts;
                            events.record(seq, Kind::PublishBegin);
                            result_pub->publish(result);
                            events.record(seq, Kind::PublishEnd);
                            ++published;
                            ++valid_count;
                            record.valid = true;
                        } catch (const std::exception &exception) {
                            if (record.call.begin && !record.call.end)
                                record.call.end = now();
                            ++invalid;
                            fail(exception.what());
                        }
                        if (record.call.returned)
                            ++operations;
                        retain(record);
                        events.record(seq, Kind::Exit);
                    });
                input_created = now();
            }
            if (role == "consumer") {
                result_sub = node->create_subscription<Result>(
                    topic + "/output", qos, [&](Result::ConstSharedPtr message) {
                        const auto entry = now();
                        ++received;
                        if (!identity.valid(*message, entry)) {
                            ++invalid;
                            fail("Invalid result identity/sample");
                            return;
                        }
                        events.record_at(message->sequence, Kind::Entry, entry);
                        const auto before_duplicates = results.duplicates;
                        const bool accepted =
                            results.accept(*message, entry, identity, fixtures);
                        if (!accepted) {
                            if (results.duplicates > before_duplicates)
                                ++duplicates;
                            else
                                ++invalid;
                            fail("Invalid or duplicate result");
                        } else {
                            ++valid_count;
                            ++observed_unique;
                        }
                        Record record;
                        record.sequence = message->sequence;
                        record.sample = message->sample_id;
                        record.generated = message->generated_ns;
                        record.entry = entry;
                        record.call.logits = message->logits;
                        record.call.returned = true;
                        record.call.output_available = true;
                        record.valid = accepted;
                        retain(record);
                        events.record(message->sequence, Kind::Exit);
                    });
                input_created = now();
            }
            std::ostringstream milestones;
            milestones << "{\"process_started_ns\":" << started
                       << ",\"model_loaded_ns\":" << model_loaded
                       << ",\"ros_initialized_ns\":" << ros_initialized
                       << ",\"node_created_ns\":" << node_created
                       << ",\"input_endpoint_returned_ns\":" << input_created
                       << ",\"output_endpoint_returned_ns\":" << output_created << '}';
            atomic_json(dir + "/" + role + ".started",
                "{\"initialized\":true,\"run_id\":" + quote(identity.run) +
                    ",\"process_id\":" + std::to_string(getpid()) +
                    ",\"inference_configuration\":" + inference_configuration +
                    ",\"milestones\":" + milestones.str() + "}");
            std::string last_ready;
            uint64_t last_ready_write = 0;
            bool startup_sampling = true;
            auto ready = [&]() {
                const auto sample_begin = now();
                const auto publishers = input_sub ? input_sub->get_publisher_count()
                    : result_sub                  ? result_sub->get_publisher_count()
                                                  : 0;
                const auto subscribers = source_pub
                    ? source_pub->get_subscription_count()
                    : result_pub ? result_pub->get_subscription_count()
                                 : 0;
                const auto sample_end = now();
                std::ostringstream out;
                out << "{\"initialized\":true,\"session_initializations\":" << sessions
                    << ",\"process_id\":" << getpid()
                    << ",\"inference_configuration\":" << inference_configuration
                    << ",\"milestones\":" << milestones.str() << ",\"input_exists\":"
                    << (input_sub || result_sub ? "true" : "false")
                    << ",\"output_exists\":"
                    << (source_pub || result_pub ? "true" : "false")
                    << ",\"warmup_count\":" << warmups
                    << ",\"warmup_begin_ns\":" << warmup_begin
                    << ",\"warmup_end_ns\":" << warmup_end
                    << ",\"matched_publishers\":" << publishers
                    << ",\"matched_subscribers\":" << subscribers << ",\"input_qos\":"
                    << (input_sub           ? qos_json(input_sub->get_actual_qos())
                               : result_sub ? qos_json(result_sub->get_actual_qos())
                                            : "null")
                    << ",\"output_qos\":"
                    << (source_pub          ? qos_json(source_pub->get_actual_qos())
                               : result_pub ? qos_json(result_pub->get_actual_qos())
                                            : "null")
                    << '}';
                if (startup_sampling && std::ifstream(dir + "/release"))
                    startup_sampling = false;
                // 启动时每 250ms 保留一个新采样；发布后仍仅在状态改变时写出。
                if (out.str() != last_ready ||
                    (startup_sampling && sample_end - last_ready_write >= second / 4)) {
                    last_ready = out.str();
                    last_ready_write = sample_end;
                    atomic_json(dir + "/" + role + ".ready.json",
                        last_ready.substr(0, last_ready.size() - 1) +
                            ",\"sample_begin_ns\":" + std::to_string(sample_begin) +
                            ",\"sample_end_ns\":" + std::to_string(sample_end) + "}");
                }
            };
            bool sent = false, acknowledged = false;
            while (rclcpp::ok() && error.empty() && now() < started + 60 * second) {
                executor.spin_once(std::chrono::milliseconds(10));
                ready();
                if (role == "source" && !sent && std::ifstream(dir + "/release")) {
                    const auto first = now();
                    schedule_start = first;
                    for (uint64_t seq = 1; seq <= identity.count && rclcpp::ok();
                         ++seq) {
                        // 绝对目标不因迟到而重置；落后时追赶，不跳过身份。
                        sleep_until(schedule.target(first, seq));
                        Input message;
                        identity.fill(message, seq);
                        message.input = fixtures.input[message.sample_id];
                        const auto generated = now();
                        message.generated_ns = generated;
                        events.record_at(seq, Kind::Generate, generated);
                        ++publish_attempts;
                        events.record(seq, Kind::PublishBegin);
                        source_pub->publish(message);
                        events.record(seq, Kind::PublishEnd);
                        ++published;
                    }
                    sent = true;
                    atomic_json(dir + "/published.json",
                        "{\"published\":" + std::to_string(published) + "}");
                }
                if (role == "consumer" && valid_count == identity.count &&
                    !acknowledged) {
                    // 完成文件在回调返回之后写出；不把它混入 Consumer 回调计时。
                    atomic_json(dir + "/consumed.json",
                        "{\"valid_unique\":" + std::to_string(valid_count) + "}");
                    acknowledged = true;
                }
            }
            if (rclcpp::ok() && error.empty())
                error = "Application safety deadline exceeded";
        } catch (const std::exception &exception) {
            error = exception.what();
        }

        // 失败也先刷新已经存在的观测，再报告真实退出状态；不补造端点。
        events.flush(
            dir + "/" + role + ".jsonl", identity.run, "source:1", role, clock);
        if (role != "source")
            flush_records(dir + "/" + role + ".results.jsonl", records, identity, clock,
                role == "processor");
        rusage usage{};
        std::string resources = "null";
        if (role == "processor" && getrusage(RUSAGE_SELF, &usage) == 0) {
            std::ostringstream out;
            out << "{\"process_peak_rss_kib\":" << usage.ru_maxrss
                << ",\"process_user_us\":"
                << usage.ru_utime.tv_sec * 1000000LL + usage.ru_utime.tv_usec
                << ",\"process_system_us\":"
                << usage.ru_stime.tv_sec * 1000000LL + usage.ru_stime.tv_usec << '}';
            resources = out.str();
        }
        std::ostringstream status;
        status << "{\"stage\":" << quote(role) << ",\"process_id\":" << getpid()
               << ",\"inference_configuration\":" << inference_configuration
               << ",\"process_resources\":" << resources
               << ",\"stored\":" << events.size() << ",\"dropped\":" << events.dropped()
               << ",\"flushed\":true"
               << ",\"clock_domain\":" << quote(clock.domain)
               << ",\"boot_id\":" << quote(clock.boot_id)
               << ",\"time_offsets\":" << quote(clock.offsets)
               << ",\"received\":" << received << ",\"published\":" << published
               << ",\"publish_attempts\":" << publish_attempts
               << ",\"observed_unique\":" << observed_unique
               << ",\"schedule_start_ns\":" << schedule_start
               << ",\"schedule_period_ns\":" << schedule.period_ns
               << ",\"inference_operations\":" << operations
               << ",\"valid_unique\":" << valid_count << ",\"invalid\":" << invalid
               << ",\"duplicates\":" << duplicates << ",\"missing_results\":"
               << (role == "consumer" ? identity.count - valid_count : 0)
               << ",\"session_initializations\":" << sessions
               << ",\"warmup_count\":" << warmups
               << ",\"warmup_begin_ns\":" << warmup_begin
               << ",\"warmup_end_ns\":" << warmup_end
               << ",\"diagnostic_dropped\":" << diagnostic_dropped
               << ",\"error\":" << quote(error) << '}';
        atomic_json(dir + "/" + role + ".status.json", status.str());
        if (rclcpp::ok())
            rclcpp::shutdown();
        if (!error.empty())
            std::cerr << role << ": " << error << '\n';
        return error.empty() ? 0 : 2;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        if (rclcpp::ok())
            rclcpp::shutdown();
        return 1;
    }
}
