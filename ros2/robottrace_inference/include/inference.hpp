#pragma once
// ROS2 固定消息的身份和计数契约；模型计算及参考张量在独立验证中复用。
#include "mobilenet.hpp"
#include "robottrace/events.hpp"
#include "robottrace_inference/msg/inference_input.hpp"
#include "robottrace_inference/msg/inference_result.hpp"

namespace inference {
    using Input = robottrace_inference::msg::InferenceInput;
    using Result = robottrace_inference::msg::InferenceResult;
    using vision::Call;
    using vision::CpuModel;
    using vision::Fixtures;
    using vision::Logits;
    using vision::sample_count;
    using vision::Tensor;
    constexpr uint64_t max_messages = 1000;

    inline int processor_threads(const std::string &value) {
        if (value == "1")
            return 1;
        if (value == "4")
            return 4;
        throw std::invalid_argument("intra-op-threads must be 1 or 4");
    }

    struct Schedule {
        uint64_t period_ns;

        Schedule(uint64_t count, double rate) {
            if (count < 1 || count > max_messages || !std::isfinite(rate) || rate < 1 ||
                rate > 10000)
                throw std::runtime_error("Invalid count/rate bounds");
            period_ns = uint64_t(double(robottrace::second) / rate);
            if ((count - 1) * period_ns > 10 * robottrace::second)
                throw std::runtime_error("Planned source window exceeds ten seconds");
        }

        uint64_t target(uint64_t first, uint64_t sequence) const {
            return first + (sequence - 1) * period_ns;
        }
    };

    struct Identity {
        std::string run;
        uint64_t hi, lo, count;

        Identity(const std::string &value, uint64_t n) : run(value), count(n) {
            if (run.size() != 32 ||
                run.find_first_not_of("0123456789abcdef") != std::string::npos ||
                n < 1 || n > max_messages)
                throw std::runtime_error("Invalid run identity or count");
            hi = std::stoull(run.substr(0, 16), nullptr, 16);
            lo = std::stoull(run.substr(16), nullptr, 16);
        }

        template <class Message> bool valid(const Message &m, uint64_t entry) const {
            return m.run_hi == hi && m.run_lo == lo && m.source_id == 1 &&
                m.sequence >= 1 && m.sequence <= count && m.generated_ns > 0 &&
                m.generated_ns <= entry &&
                m.sample_id == (m.sequence - 1) % sample_count;
        }

        template <class Message> void fill(Message &m, uint64_t sequence) const {
            m.run_hi = hi;
            m.run_lo = lo;
            m.source_id = 1;
            m.sequence = sequence;
            m.sample_id = (sequence - 1) % sample_count;
        }
    };

    struct Results {
        std::array<bool, max_messages + 1> seen{};
        uint64_t valid = 0, invalid = 0, duplicates = 0;

        bool accept(const Result &m, uint64_t entry, const Identity &identity,
            const Fixtures &fixtures) {
            if (!identity.valid(m, entry) || !fixtures.correct(m.logits, m.sample_id)) {
                ++invalid;
                return false;
            }
            if (seen[m.sequence]) {
                ++duplicates;
                return false;
            }
            seen[m.sequence] = true;
            ++valid;
            return true;
        }
    };
} // namespace inference
