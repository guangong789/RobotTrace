#pragma once
// 实验阶段控制与事件采集分离；不改变上游 String 内容或 1 Hz 定时器。
#include "robottrace/events.hpp"
#include <cstdlib>
#include <filesystem>

namespace rt_external {
    inline std::string setting(const char *name) {
        const auto value = std::getenv(name);
        if (!value || !*value)
            throw std::runtime_error(std::string("Missing setting ") + name);
        return value;
    }

    inline uint64_t number(const char *name, uint64_t maximum) {
        const auto text = setting(name);
        size_t used = 0;
        const auto value = std::stoull(text, &used);
        if (used != text.size() ||
            text.find_first_not_of("0123456789") != std::string::npos ||
            value > maximum)
            throw std::runtime_error(std::string("Invalid setting ") + name);
        return value;
    }

    inline void atomic_json(const std::string &path, const std::string &json) {
        {
            auto file = robottrace::output(path + ".tmp");
            file << json << '\n';
            file.close();
        }
        std::filesystem::rename(path + ".tmp", path);
    }

    class ExperimentalControl {
        std::string directory_ = setting("RT_OUTPUT");
        uint64_t count_ = number("RT_COUNT", 100);
        uint64_t prefix_ = number("RT_PREFIX", count_);

    public:
        bool permit(uint64_t next) const {
            return next <= count_ &&
                (std::filesystem::exists(directory_ + "/resume") ||
                    (next <= prefix_ &&
                        std::filesystem::exists(directory_ + "/prefix.enable")));
        }

        void published(uint64_t seq, size_t subscribers) const {
            if (seq == prefix_ || seq == count_)
                atomic_json(
                    directory_ + (seq == count_ ? "/published.json" : "/prefix.json"),
                    "{\"published\":" + std::to_string(seq) +
                        ",\"matched_subscribers\":" + std::to_string(subscribers) +
                        ",\"timestamp_ns\":" + std::to_string(robottrace::now()) + "}");
        }
    };
}
