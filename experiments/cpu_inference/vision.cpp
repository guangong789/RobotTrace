#include "mobilenet.hpp"
#include <charconv>
#include <cerrno>
#include <iostream>
#include <sys/resource.h>

namespace {
    constexpr size_t warmup_count = 5, measured_iterations = 100;

    struct Observation {
        size_t iteration, sample;
        uint64_t begin, end;
        bool returned, available, correct;

        Observation(
            size_t iteration, size_t sample, const vision::Call &call, bool correct)
            : iteration(iteration), sample(sample), begin(call.begin), end(call.end),
              returned(call.returned), available(call.output_available),
              correct(correct) {
        }
    };

    struct Phase {
        std::string name;
        uint64_t wall_begin = 0, cpu_begin = 0, cpu_end = 0, wall_end = 0;
        size_t calls_before = 0, calls_after = 0, valid_outputs = 0;
        bool alive_before = false, alive_after = false;
    };

    Phase begin_phase(const std::string &name, size_t calls, bool alive) {
        Phase phase;
        phase.name = name;
        phase.calls_before = calls;
        phase.alive_before = alive;
        phase.wall_begin = vision::clock_ns();
        phase.cpu_begin = vision::process_cpu_ns();
        return phase;
    }

    void end_phase(Phase &phase, size_t calls, bool alive) {
        phase.cpu_end = vision::process_cpu_ns();
        phase.wall_end = vision::clock_ns();
        phase.calls_after = calls;
        phase.alive_after = alive;
        if (phase.cpu_end < phase.cpu_begin || phase.wall_end <= phase.wall_begin)
            throw std::runtime_error("Invalid phase CPU/wall clock interval");
    }

    uint64_t resolution_ns(clockid_t clock) {
        timespec resolution{};
        if (clock_getres(clock, &resolution))
            throw std::runtime_error("Resource clock resolution unavailable");
        return uint64_t(resolution.tv_sec) * 1000000000ULL +
            uint64_t(resolution.tv_nsec);
    }

    void write_phase(const Phase &p) {
        const auto wall = p.wall_end - p.wall_begin;
        const auto cpu = p.cpu_end - p.cpu_begin;
        std::cout << "{\"name\":\"" << p.name << "\",\"wall_begin_ns\":" << p.wall_begin
                  << ",\"wall_end_ns\":" << p.wall_end
                  << ",\"cpu_begin_ns\":" << p.cpu_begin
                  << ",\"cpu_end_ns\":" << p.cpu_end << ",\"wall_ns\":" << wall
                  << ",\"cpu_ns\":" << cpu
                  << ",\"cpu_wall_ratio\":" << (wall ? double(cpu) / double(wall) : 0)
                  << ",\"run_calls_before\":" << p.calls_before
                  << ",\"run_calls_after\":" << p.calls_after
                  << ",\"valid_outputs\":" << p.valid_outputs
                  << ",\"session_alive_before\":" << (p.alive_before ? "true" : "false")
                  << ",\"session_alive_after\":" << (p.alive_after ? "true" : "false")
                  << '}';
    }

    int thread_count(const std::string &value) {
        int count = 0;
        const auto parsed =
            std::from_chars(value.data(), value.data() + value.size(), count);
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
    bool passed = false, resource_mode = false, disable_spinning = false;
    size_t run_calls = 0;
    uint64_t wall_resolution = 0, cpu_resolution = 0;
    SessionConstructionMeasurement construction;
    Phase fixture_load, warmup, active;
    std::vector<Phase> phases;
    std::vector<Observation> observations;
    observations.reserve(measured_iterations);
    try {
        const std::string usage = "Usage: vision_inference FIXTURES "
                                  "[--intra-op-num-threads N] [--resource-phases] "
                                  "[--intra-op-spinning default|disabled]";
        if (argc < 2)
            throw std::invalid_argument(usage);
        bool have_threads = false, have_spinning = false;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--resource-phases" && !resource_mode)
                resource_mode = true;
            else if (option == "--intra-op-num-threads" && !have_threads &&
                i + 1 < argc) {
                intra_op_num_threads = thread_count(argv[++i]);
                have_threads = true;
            } else if (option == "--intra-op-spinning" && !have_spinning &&
                i + 1 < argc) {
                const std::string mode = argv[++i];
                if (mode != "default" && mode != "disabled")
                    throw std::invalid_argument(
                        "intra-op-spinning must be default or disabled");
                disable_spinning = mode == "disabled";
                have_spinning = true;
            } else
                throw std::invalid_argument(usage);
        }
        if ((have_spinning && !resource_mode) ||
            (resource_mode && intra_op_num_threads != 1 && intra_op_num_threads != 4) ||
            (disable_spinning && intra_op_num_threads != 4))
            throw std::invalid_argument(
                "Resource phases require threads 1/4; disabled spinning requires 4");
        if (resource_mode) {
            wall_resolution = resolution_ns(CLOCK_MONOTONIC);
            cpu_resolution = resolution_ns(CLOCK_PROCESS_CPUTIME_ID);
            fixture_load = begin_phase("fixture_load", 0, false);
        }
        Fixtures fixtures(argv[1]);
        if (resource_mode)
            end_phase(fixture_load, 0, false);
        CpuModel model(std::string(argv[1]) + "/data/mobilenetv2-12.onnx",
            intra_op_num_threads, resource_mode ? &construction : nullptr,
            disable_spinning);
        if (resource_mode) {
            Phase session;
            session.name = "session_construction";
            session.wall_begin = construction.wall_begin;
            session.cpu_begin = construction.cpu_begin;
            session.cpu_end = construction.cpu_end;
            session.wall_end = construction.wall_end;
            session.alive_after = model.session_alive();
            phases.push_back(session);
            warmup = begin_phase("warmup", run_calls, model.session_alive());
        }
        sessions = 1;
        warmup_begin = clock_ns();
        for (size_t i = 0; i < warmup_count + measured_iterations; ++i) {
            if (resource_mode && i == warmup_count) {
                end_phase(warmup, run_calls, model.session_alive());
                phases.push_back(warmup);
                active =
                    begin_phase("active_inference", run_calls, model.session_alive());
            }
            const auto sample = i % sample_count;
            Call call;
            try {
                ++run_calls;
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
                max_error = std::max(
                    max_error, std::abs(call.logits[j] - fixtures.expected[sample][j]));
            if (resource_mode) {
                if (i < warmup_count)
                    ++warmup.valid_outputs;
                else
                    ++active.valid_outputs;
            }
            if (i < warmup_count) {
                ++warmups;
                if (warmups == warmup_count)
                    warmup_end = clock_ns();
            }
        }
        if (resource_mode) {
            end_phase(active, run_calls, model.session_alive());
            phases.push_back(active);
            auto idle =
                begin_phase("live_session_idle", run_calls, model.session_alive());
            // 同一 Session 保持存活；这两秒不调用 Run、不验证 tensor、不输出日志。
            timespec request{2, 0}, remaining{};
            int code;
            while ((code = clock_nanosleep(CLOCK_MONOTONIC, 0, &request, &remaining)) ==
                EINTR)
                request = remaining;
            if (code != 0)
                throw std::runtime_error("Live-session idle wait failed");
            end_phase(idle, run_calls, model.session_alive());
            phases.push_back(idle);
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
    std::cout
        << "{\"status\":\"" << (passed ? "passed" : "failed") << "\",\"runtime\":\""
        << Ort::GetVersionString()
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
                  << ",\"api_begin_ns\":" << o.begin << ",\"api_end_ns\":" << o.end
                  << ",\"duration_ns\":";
        if (o.returned && o.end >= o.begin)
            std::cout << o.end - o.begin;
        else
            std::cout << "null";
        std::cout << ",\"api_returned\":" << (o.returned ? "true" : "false")
                  << ",\"output_available\":" << (o.available ? "true" : "false")
                  << ",\"numerically_correct\":" << (o.correct ? "true" : "false")
                  << '}';
    }
    std::cout.precision(12);
    std::cout << "],\"resource_measurement\":";
    if (!resource_mode)
        std::cout << "null";
    else {
        std::cout << "{\"wall_clock\":\"CLOCK_MONOTONIC\","
                     "\"cpu_clock\":\"CLOCK_PROCESS_CPUTIME_ID\","
                     "\"wall_resolution_ns\":"
                  << wall_resolution << ",\"cpu_resolution_ns\":" << cpu_resolution
                  << ",\"session_creation_success\":"
                  << (construction.created ? "true" : "false")
                  << ",\"total_run_calls\":" << run_calls
                  << ",\"idle_target_ns\":2000000000,\"spinning\":{"
                     "\"key\":\"session.intra_op.allow_spinning\",\"sdk_documented_"
                     "default\":\"1\","
                     "\"explicitly_set\":"
                  << (construction.spinning_explicit ? "true" : "false")
                  << ",\"option_entry_present\":"
                  << (construction.spinning_entry_present ? "true" : "false")
                  << ",\"option_entry_value\":";
        if (construction.spinning_entry_present)
            std::cout << '"' << construction.spinning_entry << '"';
        else
            std::cout << "null";
        std::cout << "},\"fixture_load\":";
        if (fixture_load.wall_end)
            write_phase(fixture_load);
        else
            std::cout << "null";
        std::cout << ",\"phases\":[";
        for (size_t i = 0; i < phases.size(); ++i) {
            if (i)
                std::cout << ',';
            write_phase(phases[i]);
        }
        std::cout << "]}";
    }
    std::cout << ",\"process_peak_rss_kib\":" << usage.ru_maxrss
              << ",\"process_user_us\":"
              << usage.ru_utime.tv_sec * 1000000LL + usage.ru_utime.tv_usec
              << ",\"process_system_us\":"
              << usage.ru_stime.tv_sec * 1000000LL + usage.ru_stime.tv_usec << "}\n";
    return passed ? 0 : 1;
}
