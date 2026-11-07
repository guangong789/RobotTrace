#pragma once
// 不依赖 ROS 的应用事件缓冲区；预先分配容量，由单一线程独占。
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <cerrno>

namespace robottrace {
    constexpr uint64_t second = 1000000000ULL;

    inline uint64_t now() {
        timespec t{};
        if (clock_gettime(CLOCK_MONOTONIC, &t))
            throw std::runtime_error("CLOCK_MONOTONIC failed");
        return uint64_t(t.tv_sec) * second + uint64_t(t.tv_nsec);
    }

    inline void sleep_until(uint64_t target) {
        timespec t{time_t(target / second), long(target % second)};
        int rc;
        do {
            rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr);
        } while (rc == EINTR);
        if (rc)
            throw std::runtime_error("clock_nanosleep failed");
    }

    inline std::string quote(const std::string &s) {
        std::string r = "\"";
        for (unsigned char c : s) {
            if (c == '"' || c == '\\') {
                r += '\\';
                r += char(c);
            } else if (c == '\n')
                r += "\\n";
            else if (c == '\r')
                r += "\\r";
            else if (c == '\t')
                r += "\\t";
            else if (c < 32)
                r += '?';
            else
                r += char(c);
        }
        return r + '"';
    }

    inline std::string read_text(const std::string &path) {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("Cannot read " + path);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }

    inline std::ofstream output(const std::string &path) {
        std::ofstream f(path);
        if (!f)
            throw std::runtime_error("Cannot write " + path);
        f.exceptions(std::ios::failbit | std::ios::badbit);
        return f;
    }

    struct Clock {
        std::string boot_id, offsets, domain;

        Clock() {
            boot_id = read_text("/proc/sys/kernel/random/boot_id");
            if (!boot_id.empty() && boot_id.back() == '\n')
                boot_id.pop_back();
            offsets = read_text("/proc/self/timens_offsets");
            std::ifstream f("/proc/self/timens_offsets");
            std::string name;
            long long secs, nanos;
            bool mono = false, boot = false;
            while (f >> name >> secs >> nanos) {
                if (secs || nanos)
                    throw std::runtime_error(
                        "Unsupported nonzero time namespace offset");
                mono |= name == "monotonic";
                boot |= name == "boottime";
            }
            if (!mono || !boot || boot_id.size() != 36)
                throw std::runtime_error("Unsupported clock evidence");
            domain = "linux:CLOCK_MONOTONIC:" + boot_id + ":offset=0";
        }
    };
    enum class Kind { Generate, PublishBegin, PublishEnd, Entry, Exit };

    inline const char *name(Kind k) {
        switch (k) {
            case Kind::Generate:
                return "generate";
            case Kind::PublishBegin:
                return "publish_begin";
            case Kind::PublishEnd:
                return "publish_end";
            case Kind::Entry:
                return "callback_entry";
            case Kind::Exit:
                return "callback_exit";
        }
        return "invalid";
    }

    struct Observation {
        uint64_t sequence, timestamp;
        Kind kind;
    };

    class Buffer {
        std::vector<Observation> data_;
        size_t capacity_;
        uint64_t dropped_ = 0;

    public:
        explicit Buffer(size_t capacity) : capacity_(capacity) {
            data_.reserve(capacity);
        }

        bool record_at(uint64_t sequence, Kind kind, uint64_t timestamp) {
            if (data_.size() == capacity_) {
                ++dropped_;
                return false;
            }
            data_.push_back({sequence, timestamp, kind});
            return true;
        }

        bool record(uint64_t sequence, Kind kind) {
            return record_at(sequence, kind, now());
        }

        size_t size() const {
            return data_.size();
        }

        uint64_t dropped() const {
            return dropped_;
        }

        void flush(const std::string &path, const std::string &run,
            const std::string &source, const std::string &stage,
            const Clock &clock) const {
            auto f = output(path);
            for (const auto &e : data_)
                f << "{\"schema_version\":1,\"run_id\":" << quote(run)
                  << ",\"source_id\":" << quote(source)
                  << ",\"sequence\":" << e.sequence << ",\"stage\":" << quote(stage)
                  << ",\"event_type\":" << quote(name(e.kind))
                  << ",\"timestamp_ns\":" << e.timestamp
                  << ",\"clock_domain\":" << quote(clock.domain)
                  << ",\"process_id\":" << getpid() << "}\n";
            f.flush(); // 成功返回只表示正常进程已完成输出，
                       // 不保证崩溃后的数据持久性。
        }
    };
}
