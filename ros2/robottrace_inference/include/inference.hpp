#pragma once
// ROS2 固定消息的身份和计数契约；模型计算及参考张量在独立验证中复用。
#include "mobilenet.hpp"
#include "robottrace/events.hpp"
#include "robottrace_inference/msg/inference_input.hpp"
#include "robottrace_inference/msg/inference_result.hpp"
#include <filesystem>
#include <set>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

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

    struct ProcessorOptions {
        std::string provider = "cpu";
        int threads = 1;
        bool warmup_only = false;
    };

    inline ProcessorOptions processor_options(const std::string &role,
        const std::vector<std::string> &arguments, bool cuda_build) {
        ProcessorOptions result;
        std::set<std::string> seen;
        for (size_t i = 0; i < arguments.size(); ++i) {
            const auto &key = arguments[i];
            if (!seen.insert(key).second)
                throw std::invalid_argument("Duplicate Processor option");
            if (key == "--warmup-only") {
                result.warmup_only = true;
                continue;
            }
            if (i + 1 == arguments.size())
                throw std::invalid_argument("Missing Processor option value");
            const auto &value = arguments[++i];
            if (key == "--provider") {
                if (value != "cpu" && value != "cuda")
                    throw std::invalid_argument("provider must be cpu or cuda");
                result.provider = value;
            } else if (key == "--intra-op-threads" && role == "processor")
                result.threads = processor_threads(value);
            else
                throw std::invalid_argument("Unsupported role/Processor option");
        }
        if (cuda_build && (role != "processor" || result.provider != "cuda" ||
                             !seen.count("--provider") || result.threads != 1))
            throw std::invalid_argument("CUDA build requires Processor --provider cuda and one host thread");
        if (!cuda_build && (result.provider != "cpu" || result.warmup_only))
            throw std::invalid_argument("CPU build accepts only CPU execution");
        return result;
    }

    inline std::string cuda_warmup_gate(const std::string &dir, const std::string &profile) {
        const auto libraries_path = dir + "/processor.ort-libraries.json";
        auto file = robottrace::output(libraries_path);
        std::ifstream maps("/proc/self/maps");
        if (!maps)
            throw std::runtime_error("Cannot inspect loaded ORT libraries");
        std::set<std::string> paths;
        std::string line;
        while (std::getline(maps, line)) {
            const auto start = line.find('/');
            if (start != std::string::npos &&
                (line.find("/libonnxruntime.so") != std::string::npos ||
                    line.find("/libonnxruntime_providers_") != std::string::npos))
                paths.insert(line.substr(start));
        }
        file << '[';
        bool comma = false;
        for (const auto &path : paths) {
            if (comma) file << ',';
            file << robottrace::quote(path);
            comma = true;
        }
        file << "]\n";
        file.close();
        const auto pid_text = std::to_string(getpid());
        const auto result_path = dir + "/processor.cuda-execution.json";
        const char *arguments[]{"python3", "-B", "-m", "robottrace.inference_cuda",
            "verify-profile", profile.c_str(), "--pid", pid_text.c_str(),
            "--libraries", libraries_path.c_str(), "--sdk-root", ROBOTTRACE_ORT_ROOT,
            "--output", result_path.c_str(), nullptr};
        std::vector<char *> command;
        for (auto argument : arguments) command.push_back(const_cast<char *>(argument));
        pid_t child = 0;
        if (posix_spawnp(&child, "python3", nullptr, nullptr, command.data(), ::environ))
            throw std::runtime_error("Cannot start CUDA evidence validator");
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("CUDA warm-up execution evidence rejected");
        return robottrace::read_text(result_path);
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
}
