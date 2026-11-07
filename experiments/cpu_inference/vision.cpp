// 独立验证真实照片与完整参考输出，记录同步推理 API 的原始观测。
#include "mobilenet.hpp"
#include <charconv>
#include <iostream>
#include <sys/resource.h>

namespace {
    constexpr size_t warmup_count = 5, measured_iterations = 100;

    struct Observation {
        size_t iteration, sample;
        uint64_t begin, end;
        bool returned, available, correct;

        Observation(size_t iteration, size_t sample, const vision::Call &call, bool correct)
            : iteration(iteration), sample(sample), begin(call.begin), end(call.end),
              returned(call.returned), available(call.output_available), correct(correct) {}
    };

    int thread_count(const std::string &value) {
        int count = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            (count != 1 && count != 2 && count != 4))
            throw std::invalid_argument(
                "intra_op_num_threads must be a positive integer from {1, 2, 4}");
        return count;
    }
}

int main(int argc, char **argv) {
    using namespace vision;
    int intra_op_num_threads = 1;
    size_t warmups = 0, sessions = 0;
    uint64_t warmup_begin = 0, warmup_end = 0;
    float max_error = 0;
    bool passed = false;
    std::vector<Observation> observations;
    observations.reserve(measured_iterations);
    try {
        if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--intra-op-num-threads"))
            throw std::invalid_argument(
                "Usage: vision_inference FIXTURES [--intra-op-num-threads N]");
        if (argc == 4)
            intra_op_num_threads = thread_count(argv[3]);
        Fixtures fixtures(argv[1]);
        CpuModel model(std::string(argv[1]) + "/data/mobilenetv2-12.onnx", intra_op_num_threads);
        sessions = 1;
        warmup_begin = clock_ns();
        for (size_t i = 0; i < warmup_count + measured_iterations; ++i) {
            const auto sample = i % sample_count;
            Call call;
            try {
                model.run(fixtures.input[sample], call);
            } catch (const std::exception &) {
                // 失败调用保留已有时间戳；未返回的调用不伪造完整 duration。
                if (i >= warmup_count)
                    observations.push_back({i - warmup_count + 1, sample, call, false});
                throw;
            }
            const bool correct = call.returned && call.output_available &&
                fixtures.correct(call.logits, sample);
            if (i >= warmup_count)
                observations.push_back({i - warmup_count + 1, sample, call, correct});
            if (!correct)
                throw std::runtime_error("Independent full-output reference mismatch");
            for (size_t j = 0; j < output_elements; ++j)
                max_error = std::max(max_error,
                    std::abs(call.logits[j] - fixtures.expected[sample][j]));
            if (i < warmup_count) {
                ++warmups;
                if (warmups == warmup_count)
                    warmup_end = clock_ns();
            }
        }
        passed = true;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }

    // 统计与输出全部位于 API 计时之外；失败与不完整观测不会成为成功结果。
    std::vector<uint64_t> ordered;
    for (const auto &observation : observations)
        if (observation.correct)
            ordered.push_back(observation.end - observation.begin);
    std::sort(ordered.begin(), ordered.end());
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) {
        passed = false;
        std::cerr << "FAIL: getrusage failed\n";
    }
    std::cout << "{\"status\":\"" << (passed ? "passed" : "failed")
              << "\",\"runtime\":\"" << Ort::GetVersionString()
              << "\",\"provider\":\"CPUExecutionProvider\",\"intra_op_num_threads\":"
              << intra_op_num_threads
              << ",\"inter_op_num_threads\":1,\"execution_mode\":\"sequential\","
                 "\"graph_optimization\":\"basic\",\"cpu_arena\":true,"
                 "\"clock\":\"CLOCK_MONOTONIC\",\"session_initializations\":"
              << sessions << ",\"warmup_count\":" << warmups
              << ",\"expected_warmup_count\":" << warmup_count
              << ",\"expected_measured_iterations\":" << measured_iterations
              << ",\"sample_count\":2,\"input_elements\":150528,\"output_elements\":1000,"
                 "\"warmup_begin_ns\":"
              << warmup_begin << ",\"warmup_end_ns\":" << warmup_end
              << ",\"atol\":" << vision::atol << ",\"rtol\":" << rtol
              << ",\"max_abs_error\":" << max_error
              << ",\"api_ns\":{\"count\":" << ordered.size();
    if (ordered.empty())
        std::cout << ",\"p50\":null,\"p95\":null,\"min\":null,\"max\":null";
    else
        std::cout << ",\"p50\":" << ordered[(ordered.size() + 1) / 2 - 1]
                  << ",\"p95\":" << ordered[(95 * ordered.size() + 99) / 100 - 1]
                  << ",\"min\":" << ordered.front() << ",\"max\":" << ordered.back();
    std::cout << "},\"observations\":[";
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto &o = observations[i];
        if (i)
            std::cout << ',';
        std::cout << "{\"iteration\":" << o.iteration << ",\"sample_id\":" << o.sample
                  << ",\"api_begin_ns\":" << o.begin
                  << ",\"api_end_ns\":" << o.end << ",\"duration_ns\":";
        if (o.returned && o.end >= o.begin)
            std::cout << o.end - o.begin;
        else
            std::cout << "null";
        std::cout << ",\"api_returned\":" << (o.returned ? "true" : "false")
                  << ",\"output_available\":" << (o.available ? "true" : "false")
                  << ",\"numerically_correct\":" << (o.correct ? "true" : "false") << '}';
    }
    std::cout << "],\"process_peak_rss_kib\":" << usage.ru_maxrss
              << ",\"process_user_us\":"
              << usage.ru_utime.tv_sec * 1000000LL + usage.ru_utime.tv_usec
              << ",\"process_system_us\":"
              << usage.ru_stime.tv_sec * 1000000LL + usage.ru_stime.tv_usec << "}\n";
    return passed ? 0 : 1;
}
