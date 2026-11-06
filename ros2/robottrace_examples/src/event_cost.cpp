#include "robottrace/events.hpp"
#include <algorithm>
#include <iostream>

int main() {
    using namespace robottrace;
    Clock clock;
    constexpr size_t count = 20000;
    std::vector<double> costs;
    for (int round = 0; round < 7; ++round) {
        Buffer buffer(count);
        const auto begin = now();
        for (size_t i = 0; i < count; ++i)
            buffer.record(i, Kind::Entry);
        const auto end = now();
        if (buffer.size() != count || buffer.dropped())
            return 1;
        costs.push_back(double(end - begin) / count);
        // 在计时区间结束后，通过可观测的 I/O 读取所有已存观测。
        // 这样可以防止优化器删除
        // 未被使用的事件数据写入。
        buffer.flush("/dev/null", "overhead", "source:1", "consumer", clock);
    }
    std::sort(costs.begin(), costs.end());
    Buffer tiny(1);
    tiny.record(1, Kind::Entry);
    tiny.record(1, Kind::Exit);
    if (tiny.size() != 1 || tiny.dropped() != 1)
        return 2;
    std::cout
        << "{\"measurement\":\"CLOCK_MONOTONIC sample plus push into pre-reserved "
           "single-owner buffer; excludes reserve, JSON formatting, file I/O and "
           "ROS\",\"rounds\":7,\"events_per_round\":20000,\"ns_per_event_round_mean\":{"
           "\"min\":"
        << costs.front() << ",\"median\":" << costs[3] << ",\"max\":" << costs.back()
        << "},\"overflow_probe\":{\"capacity\":1,\"stored\":1,\"dropped\":1}}\n";
}
