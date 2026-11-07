#pragma once
// RobotTrace 增补；只连接本次上游应用与现有 Buffer/Clock，不提供通用 SDK。
#include "identity.hpp"
#include "experimental_control.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <iostream>

namespace rt_external {
    inline std::string qos(const rclcpp::QoS &value) {
        const auto &q = value.get_rmw_qos_profile();
        return "{\"history\":" + std::to_string(q.history) +
            ",\"depth\":" + std::to_string(q.depth) +
            ",\"reliability\":" + std::to_string(q.reliability) +
            ",\"durability\":" + std::to_string(q.durability) + "}";
    }

    class Trace {
        std::string stage_, directory_ = setting("RT_OUTPUT"), run_ = setting("RT_RUN");
        std::string source_ = "demo:talker";
        uint64_t count_ = number("RT_COUNT", 100), invalid_ = 0;
        Identities identities_{count_};
        robottrace::Clock clock_;
        robottrace::Buffer buffer_{number("RT_CAPACITY", 1000000)};
        std::string last_ready_, error_;

        std::string clock_json() const {
            return "\"stage\":" + robottrace::quote(stage_) +
                ",\"process_id\":" + std::to_string(getpid()) +
                ",\"clock_domain\":" + robottrace::quote(clock_.domain) +
                ",\"boot_id\":" + robottrace::quote(clock_.boot_id) +
                ",\"time_offsets\":" + robottrace::quote(clock_.offsets);
        }

    public:
        explicit Trace(const std::string &stage) : stage_(stage) {
            if (run_.size() != 32 ||
                run_.find_first_not_of("0123456789abcdef") != std::string::npos ||
                !count_)
                throw std::runtime_error("Invalid experiment identity/count");
            atomic_json(
                directory_ + "/" + stage_ + ".started", "{" + clock_json() + "}");
        }

        ~Trace() noexcept {
            try {
                buffer_.flush(directory_ + "/" + stage_ + ".jsonl", run_, source_,
                    stage_, clock_);
                atomic_json(directory_ + "/" + stage_ + ".status.json",
                    "{" + clock_json() +
                        ",\"stored\":" + std::to_string(buffer_.size()) +
                        ",\"dropped\":" + std::to_string(buffer_.dropped()) +
                        ",\"flushed\":true" +
                        ",\"invalid_messages\":" + std::to_string(invalid_) +
                        ",\"identity_error\":" + robottrace::quote(error_) + "}");
            } catch (const std::exception &error) {
                // 状态文件缺失会让运行器明确判为失败，不伪造成功刷新。
                std::cerr << "RobotTrace flush failed: " << error.what() << std::endl;
            }
        }

        uint64_t accept(const std::string &text) {
            try {
                return identities_.accept(text);
            } catch (const std::invalid_argument &error) {
                reject(error.what());
                return 0;
            }
        }

        void reject(const std::string &error) {
            ++invalid_;
            error_ = error;
        }

        void record(uint64_t seq, robottrace::Kind kind) {
            buffer_.record(seq, kind);
        }

        void record_at(uint64_t seq, robottrace::Kind kind, uint64_t stamp) {
            buffer_.record_at(seq, kind, stamp);
        }

        void ready(size_t publishers, size_t subscribers,
            const std::string &input_qos = "null",
            const std::string &output_qos = "null") {
            const auto state = "{\"matched_publishers\":" + std::to_string(publishers) +
                ",\"matched_subscribers\":" + std::to_string(subscribers) +
                ",\"input_qos\":" + input_qos + ",\"output_qos\":" + output_qos + "}";
            if (state != last_ready_) {
                atomic_json(directory_ + "/" + stage_ + ".ready.json", state);
                last_ready_ = state;
            }
        }

        void delivered(uint64_t seq) const {
            if (seq == count_)
                atomic_json(directory_ + "/delivered.json",
                    "{\"last_sequence\":" + std::to_string(seq) +
                        ",\"timestamp_ns\":" + std::to_string(robottrace::now()) + "}");
        }
    };
}
