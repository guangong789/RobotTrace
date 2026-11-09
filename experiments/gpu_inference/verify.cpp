#include "../cpu_inference/mobilenet.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>

namespace {
    std::set<std::string> loaded_libraries() {
        std::ifstream maps("/proc/self/maps");
        if (!maps)
            throw std::runtime_error("Cannot inspect process library mappings");
        std::set<std::string> paths;
        std::string line;
        while (std::getline(maps, line)) {
            const auto start = line.find('/');
            if (start != std::string::npos &&
                (line.find("onnxruntime") != std::string::npos ||
                    line.find("libcuda") != std::string::npos ||
                    line.find("libcublas") != std::string::npos ||
                    line.find("libcufft") != std::string::npos ||
                    line.find("libcurand") != std::string::npos ||
                    line.find("libnvrtc") != std::string::npos))
                paths.insert(line.substr(start));
        }
        return paths;
    }

    void write_paths(const std::set<std::string> &paths) {
        bool comma = false;
        std::cout << '[';
        for (const auto &path : paths) {
            if (comma)
                std::cout << ',';
            std::cout << std::quoted(path);
            comma = true;
        }
        std::cout << ']';
    }
}

int main(int argc, char **argv) {
    using namespace vision;
    StandaloneSessionOptions settings;
    std::string expected_library, profile_path;
    bool passed = false, library_verified = false;
    std::set<std::string> libraries;
    struct Observation {
        size_t sample;
        Call call;
        bool correct;
    };
    std::vector<Observation> observations;
    try {
        const std::string usage = "Usage: verify_cuda FIXTURES "
                                  "--expect-ort-library PATH "
                                  "[--provider cpu|cuda] [--profile-prefix PATH]";
        if (argc < 2)
            throw std::invalid_argument(usage);
        std::set<std::string> seen;
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            if (!seen.insert(key).second || i + 1 == argc)
                throw std::invalid_argument(usage);
            const std::string value = argv[++i];
            if (key == "--provider") {
                if (value != "cpu" && value != "cuda")
                    throw std::invalid_argument("provider must be cpu or cuda");
                settings.provider = value;
            } else if (key == "--expect-ort-library")
                expected_library = value;
            else if (key == "--profile-prefix")
                settings.profile_prefix = value;
            else
                throw std::invalid_argument(usage);
        }
        if (expected_library.empty() ||
            (settings.provider == "cuda" && settings.profile_prefix.empty()))
            throw std::invalid_argument(usage);
        libraries = loaded_libraries();
        const auto expected = std::filesystem::canonical(expected_library).string();
        if (!libraries.count(expected))
            throw std::runtime_error("Loaded ORT library differs from explicit expectation");
        for (const auto &path : libraries)
            if (path.find("/libonnxruntime.so") != std::string::npos && path != expected)
                throw std::runtime_error("Unexpected additional ORT main library");
        library_verified = true;
        Fixtures fixtures(argv[1]);
        CpuModel model(std::string(argv[1]) + "/data/mobilenetv2-12.onnx", 1,
            nullptr, false, &settings);
        for (size_t i = 0; i < 4; ++i) {
            const size_t sample = i % sample_count;
            Call call;
            model.run(fixtures.input[sample], call);
            const bool correct = fixtures.correct(call.logits, sample);
            observations.push_back({sample, call, correct});
            if (!correct) {
                for (size_t j = 0; j < output_elements; ++j) {
                    const auto actual = call.logits[j];
                    const auto reference = fixtures.expected[sample][j];
                    if (!std::isfinite(actual) ||
                        std::abs(actual - reference) > vision::atol + rtol * std::abs(reference))
                        std::cerr << std::setprecision(10) << "MISMATCH sample="
                                  << sample << " index=" << j << " actual=" << actual
                                  << " reference=" << reference << " tolerance="
                                  << vision::atol + rtol * std::abs(reference) << '\n';
                }
                throw std::runtime_error("Independent full-output reference mismatch");
            }
        }
        if (!settings.profile_prefix.empty())
            profile_path = model.finish_profile();
        libraries = loaded_libraries();
        passed = true;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    std::cout << std::setprecision(10)
              << "{\"status\":\"" << (passed ? "passed" : "failed")
              << "\",\"runtime\":" << std::quoted(Ort::GetVersionString())
              << ",\"provider\":" << std::quoted(settings.provider)
              << ",\"library_verified\":" << (library_verified ? "true" : "false")
              << ",\"provider_registered\":"
              << (settings.provider_registered ? "true" : "false")
              << ",\"session_created\":" << (settings.session_created ? "true" : "false")
              << ",\"cuda_options\":" << std::quoted(settings.cuda_options)
              << ",\"cpu_fallback_policy\":\"allowed and must be disclosed from profile\""
              << ",\"intra_op_num_threads\":1,\"inter_op_num_threads\":1"
              << ",\"execution_mode\":\"sequential\",\"graph_optimization\":\"basic\""
              << ",\"clock\":\"CLOCK_MONOTONIC\",\"timing\":\"host synchronous Run including transfers; diagnostic only\""
              << ",\"warmup_count\":0,\"output_elements\":1000,\"atol\":" << vision::atol
              << ",\"rtol\":" << rtol << ",\"profile_path\":" << std::quoted(profile_path)
              << ",\"loaded_libraries\":";
    write_paths(libraries);
    std::cout << ",\"observations\":[";
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto &o = observations[i];
        if (i)
            std::cout << ',';
        std::cout << "{\"sample_id\":" << o.sample << ",\"api_begin_ns\":" << o.call.begin
                  << ",\"api_end_ns\":" << o.call.end
                  << ",\"duration_ns\":" << o.call.end - o.call.begin
                  << ",\"numerically_correct\":" << (o.correct ? "true" : "false")
                  << ",\"logits\":[";
        for (size_t j = 0; j < output_elements; ++j) {
            if (j)
                std::cout << ',';
            if (std::isfinite(o.call.logits[j]))
                std::cout << o.call.logits[j];
            else
                std::cout << "null";
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
    return passed ? 0 : 1;
}
