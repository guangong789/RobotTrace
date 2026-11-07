#pragma once
// RobotTrace 的限定适配：只接受官方 Talker 的规范十进制 String 载荷。
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rt_external {
    inline uint64_t sequence(const std::string &text, uint64_t count) {
        const std::string prefix = "Hello World: ";
        if (text.compare(0, prefix.size(), prefix) || text.size() <= prefix.size())
            throw std::invalid_argument("Invalid Talker identity format");
        const char *first = text.data() + prefix.size();
        const char *last = text.data() + text.size();
        uint64_t value = 0;
        const auto parsed = std::from_chars(first, last, value);
        if (*first == '0' || parsed.ec != std::errc{} || parsed.ptr != last ||
            value == 0 || value > count)
            throw std::invalid_argument("Invalid or unexpected Talker sequence");
        return value;
    }

    class Identities {
        std::vector<bool> seen_;

    public:
        explicit Identities(uint64_t count) : seen_(count + 1, false) {
        }

        uint64_t accept(const std::string &text) {
            const auto value = sequence(text, seen_.size() - 1);
            if (seen_[value])
                throw std::invalid_argument("Duplicate Talker sequence");
            seen_[value] = true;
            return value;
        }
    };
}
