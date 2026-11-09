#include "../cpu_inference/mobilenet.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <unistd.h>

namespace {
    struct Numeric {
        size_t checked = 0, mismatches = 0, nonfinite = 0;
        double max_abs_error = 0;
        bool correct = false;
    };

    struct Observation {
        bool warmup;
        size_t index, sample;
        vision::Call call;
        Numeric numeric;
    };

    std::set<std::string> loaded_libraries() {
        std::ifstream maps("/proc/self/maps");
        if (!maps)
            throw std::runtime_error("Cannot inspect loaded libraries");
        std::set<std::string> paths;
        std::string line;
        while (std::getline(maps, line)) {
            const auto start = line.find('/');
            if (start != std::string::npos &&
                (line.find("onnxruntime") != std::string::npos ||
                    line.find("libcuda") != std::string::npos ||
                    line.find("libcudnn") != std::string::npos ||
                    line.find("libcublas") != std::string::npos ||
                    line.find("libcufft") != std::string::npos ||
                    line.find("libcurand") != std::string::npos ||
                    line.find("libnvrtc") != std::string::npos))
                paths.insert(line.substr(start));
        }
        return paths;
    }

    Numeric validate(const vision::Fixtures &fixtures, const vision::Call &call,
        size_t sample) {
        Numeric value;
        value.correct = call.returned && call.output_available &&
            fixtures.correct(call.logits, sample);
        if (!call.output_available)
            return value;
        for (size_t index = 0; index < vision::output_elements; ++index) {
            ++value.checked;
            const float actual = call.logits[index], expected = fixtures.expected[sample][index];
            const float tolerance = vision::atol + vision::rtol * std::abs(expected);
            if (!std::isfinite(actual))
                ++value.nonfinite;
            else
                value.max_abs_error = std::max(value.max_abs_error,
                    std::abs(double(actual) - double(expected)));
            if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                ++value.mismatches;
                std::cerr << std::setprecision(10) << "MISMATCH sample=" << sample
                          << " index=" << index << " actual=" << actual
                          << " reference=" << expected << " tolerance=" << tolerance << '\n';
            }
        }
        return value;
    }

    void interval(uint64_t begin, uint64_t end) {
        if (begin && end >= begin)
            std::cout << end - begin;
        else
            std::cout << "null";
    }
}

int main(int argc, char **argv) {
    vision::StandaloneSessionOptions settings;
    vision::SessionConstructionMeasurement construction;
    int threads = 1, applied_threads = 0;
    bool check = false, passed = false, library_verified = false;
    std::string expected_library, profile_path;
    uint64_t model_begin = 0, model_end = 0, warmup_begin = 0, warmup_end = 0,
             measured_begin = 0, measured_end = 0;
    std::set<std::string> libraries;
    std::vector<Observation> observations;
    observations.reserve(105);
    try {
        const std::string usage = "Usage: benchmark_inference FIXTURES "
                                  "--expect-ort-library PATH [--provider cpu|cuda] "
                                  "[--intra-op-threads 1|4] [--check] [--profile-prefix PATH]";
        if (argc < 2)
            throw std::invalid_argument(usage);
        std::set<std::string> seen;
        for (int index = 2; index < argc; ++index) {
            const std::string key = argv[index];
            if (!seen.insert(key).second)
                throw std::invalid_argument(usage);
            if (key == "--check") {
                check = true;
                continue;
            }
            if (index + 1 == argc)
                throw std::invalid_argument(usage);
            const std::string value = argv[++index];
            if (key == "--provider") {
                if (value != "cpu" && value != "cuda")
                    throw std::invalid_argument("provider must be cpu or cuda");
                settings.provider = value;
            } else if (key == "--intra-op-threads") {
                if (value != "1" && value != "4")
                    throw std::invalid_argument("intra-op threads must be 1 or 4");
                threads = value == "1" ? 1 : 4;
            } else if (key == "--expect-ort-library")
                expected_library = value;
            else if (key == "--profile-prefix")
                settings.profile_prefix = value;
            else
                throw std::invalid_argument(usage);
        }
        if (expected_library.empty() ||
            (!settings.profile_prefix.empty() && (!check || settings.provider != "cuda")) ||
            (settings.provider == "cuda" && threads != 1))
            throw std::invalid_argument(usage);
        libraries = loaded_libraries();
        const auto expected = std::filesystem::canonical(expected_library).string();
        if (!libraries.count(expected))
            throw std::runtime_error("Loaded ORT library differs from expectation");
        for (const auto &path : libraries)
            if (path.find("/libonnxruntime.so") != std::string::npos && path != expected)
                throw std::runtime_error("Additional ORT main library loaded");
        library_verified = true;
        vision::Fixtures fixtures(argv[1]);
        model_begin = vision::clock_ns();
        vision::CpuModel model(std::string(argv[1]) + "/data/mobilenetv2-12.onnx",
            threads, &construction, false, &settings);
        model_end = vision::clock_ns();
        applied_threads = model.intra_op_num_threads();
        const size_t warmups = check ? 0 : 5, measured = check ? 4 : 100;
        warmup_begin = vision::clock_ns();
        if (!warmups)
            warmup_end = warmup_begin;
        for (size_t index = 0; index < warmups + measured; ++index) {
            if (index == warmups) {
                if (warmups)
                    warmup_end = vision::clock_ns();
                measured_begin = vision::clock_ns();
            }
            Observation observation{index < warmups,
                index < warmups ? index : index - warmups, index % vision::sample_count, {}, {}};
            try {
                model.run(fixtures.input[observation.sample], observation.call);
            } catch (const std::exception &) {
                observations.push_back(observation);
                throw;
            }
            observation.numeric = validate(fixtures, observation.call, observation.sample);
            observations.push_back(observation);
            if (!observation.numeric.correct)
                throw std::runtime_error("Full-output numerical verification failed");
        }
        measured_end = vision::clock_ns();
        if (!settings.profile_prefix.empty())
            profile_path = model.finish_profile();
        libraries = loaded_libraries();
        passed = true;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    std::vector<uint64_t> ordered;
    size_t warmups_completed = 0;
    double max_error = 0;
    for (const auto &o : observations) {
        if (o.warmup && o.numeric.correct)
            ++warmups_completed;
        if (!o.warmup && o.numeric.correct)
            ordered.push_back(o.call.end - o.call.begin);
        max_error = std::max(max_error, o.numeric.max_abs_error);
    }
    std::sort(ordered.begin(), ordered.end());
    std::cout << std::setprecision(12)
              << "{\"status\":\"" << (passed ? "passed" : "failed")
              << "\",\"mode\":\"" << (check ? "check" : "benchmark")
              << "\",\"pid\":" << getpid() << ",\"runtime\":" << std::quoted(Ort::GetVersionString())
              << ",\"provider\":" << std::quoted(settings.provider)
              << ",\"intra_op_num_threads\":" << applied_threads
              << ",\"inter_op_num_threads\":1,\"execution_mode\":\"sequential\""
              << ",\"graph_optimization\":\"basic\",\"cpu_arena\":true"
              << ",\"spinning_explicitly_set\":false,\"clock\":\"CLOCK_MONOTONIC\""
              << ",\"host_input_output\":true,\"profiling_enabled\":"
              << (!settings.profile_prefix.empty() ? "true" : "false")
              << ",\"library_verified\":" << (library_verified ? "true" : "false")
              << ",\"provider_registered\":" << (settings.provider_registered ? "true" : "false")
              << ",\"session_initializations\":" << (settings.session_created ? 1 : 0)
              << ",\"cuda_options\":" << std::quoted(settings.cuda_options)
              << ",\"profile_path\":" << std::quoted(profile_path)
              << ",\"session_construction_ns\":";
    interval(construction.wall_begin, construction.wall_end);
    std::cout << ",\"model_initialization_ns\":";
    interval(model_begin, model_end);
    std::cout << ",\"warmup_count\":" << warmups_completed << ",\"warmup_wall_ns\":";
    interval(warmup_begin, warmup_end);
    std::cout << ",\"measured_begin_ns\":" << measured_begin
              << ",\"measured_end_ns\":" << measured_end << ",\"measured_wall_ns\":";
    interval(measured_begin, measured_end);
    std::cout << ",\"valid_inferences\":" << ordered.size()
              << ",\"atol\":" << vision::atol << ",\"rtol\":" << vision::rtol
              << ",\"max_abs_error\":" << max_error << ",\"api_ns\":{\"count\":" << ordered.size();
    if (ordered.empty())
        std::cout << ",\"p50\":null,\"p95\":null,\"min\":null,\"max\":null";
    else
        std::cout << ",\"p50\":" << ordered[(ordered.size() + 1) / 2 - 1]
                  << ",\"p95\":" << ordered[(95 * ordered.size() + 99) / 100 - 1]
                  << ",\"min\":" << ordered.front() << ",\"max\":" << ordered.back();
    std::cout << "},\"loaded_libraries\":[";
    bool comma = false;
    for (const auto &path : libraries) {
        if (comma)
            std::cout << ',';
        std::cout << std::quoted(path);
        comma = true;
    }
    std::cout << "],\"observations\":[";
    for (size_t index = 0; index < observations.size(); ++index) {
        const auto &o = observations[index];
        if (index)
            std::cout << ',';
        std::cout << "{\"warmup\":" << (o.warmup ? "true" : "false")
                  << ",\"index\":" << o.index << ",\"sample_id\":" << o.sample
                  << ",\"api_begin_ns\":" << o.call.begin << ",\"api_end_ns\":" << o.call.end
                  << ",\"duration_ns\":";
        interval(o.call.begin, o.call.end);
        std::cout << ",\"api_returned\":" << (o.call.returned ? "true" : "false")
                  << ",\"output_values_checked\":" << o.numeric.checked
                  << ",\"numerically_correct\":" << (o.numeric.correct ? "true" : "false")
                  << ",\"mismatches\":" << o.numeric.mismatches
                  << ",\"nonfinite\":" << o.numeric.nonfinite
                  << ",\"max_abs_error\":" << o.numeric.max_abs_error << '}';
    }
    std::cout << "]}\n";
    return passed ? 0 : 1;
}
