#pragma once
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <onnxruntime_session_options_config_keys.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

namespace vision {
    constexpr size_t sample_count = 2, input_elements = 3 * 224 * 224,
                     output_elements = 1000;
    constexpr float atol = 1e-4f, rtol = 1e-4f;
    using Tensor = std::array<float, input_elements>;
    using Logits = std::array<float, output_elements>;

    inline uint64_t clock_ns() {
        timespec value{};
        if (clock_gettime(CLOCK_MONOTONIC, &value))
            throw std::runtime_error("CLOCK_MONOTONIC failed");
        return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
    }

    template <size_t N>
    void read_tensor(const std::string &path, std::array<float, N> &out) {
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4)
            throw std::runtime_error("Expected little-endian FP32 host");
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char *>(out.data()), sizeof(out));
        if (!file || file.peek() != std::char_traits<char>::eof() ||
            std::any_of(out.begin(), out.end(), [](float f) {
                return !std::isfinite(f);
            }))
            throw std::runtime_error("Invalid fixed tensor: " + path);
    }

    struct Fixtures {
        std::array<Tensor, sample_count> input{};
        std::array<Logits, sample_count> expected{};

        explicit Fixtures(const std::string &root) {
            for (size_t i = 0; i < sample_count; ++i) {
                read_tensor(
                    root + "/data/input-" + std::to_string(i) + ".f32", input[i]);
                read_tensor(
                    root + "/data/expected-" + std::to_string(i) + ".f32", expected[i]);
            }
            if (!std::memcmp(input[0].data(), input[1].data(), sizeof(Tensor)))
                throw std::runtime_error("Duplicate input tensors");
        }

        bool correct(const Logits &values, uint32_t sample) const {
            if (sample >= sample_count)
                return false;
            for (size_t i = 0; i < output_elements; ++i)
                if (!std::isfinite(values[i]) ||
                    std::abs(values[i] - expected[sample][i]) >
                        atol + rtol * std::abs(expected[sample][i]))
                    return false;
            return true;
        }

        template <class Message> bool payload(const Message &message) const {
            return message.sample_id < sample_count &&
                !std::memcmp(message.input.data(), input[message.sample_id].data(),
                    sizeof(Tensor));
        }
    };

    struct Call {
        uint64_t begin = 0, end = 0;
        bool returned = false, output_available = false;
        Logits logits{};
    };

    // 仅 standalone 资源模式传入；默认调用不读 CPU 时钟或改变 spinning。
    inline uint64_t process_cpu_ns() {
        timespec value{};
        if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value))
            throw std::runtime_error("CLOCK_PROCESS_CPUTIME_ID failed");
        return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
    }

    struct SessionConstructionMeasurement {
        uint64_t wall_begin = 0, cpu_begin = 0, cpu_end = 0, wall_end = 0;
        bool created = false, spinning_explicit = false, spinning_entry_present = false;
        std::string spinning_entry;
    };

    // 仅由独立验证入口传入；省略时保留原有 CPU 配置和计时行为。
    struct StandaloneSessionOptions {
        std::string provider = "cpu", profile_prefix, cuda_options;
        bool provider_registered = false, session_created = false;
    };

    class CpuModel {
        Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "robottrace-vision"};
        std::unique_ptr<Ort::Session> session_;
        int applied_intra_op_num_threads_ = 0;

    public:
        explicit CpuModel(const std::string &path, int intra_op_num_threads = 1,
            SessionConstructionMeasurement *measurement = nullptr,
            bool disable_intra_op_spinning = false,
            StandaloneSessionOptions *standalone = nullptr) {
            if (intra_op_num_threads <= 0)
                throw std::invalid_argument("intra_op_num_threads must be positive");
            if (disable_intra_op_spinning && !measurement)
                throw std::invalid_argument("Spinning experiment requires resource phases");
            Ort::SessionOptions options;
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            options.SetIntraOpNumThreads(intra_op_num_threads);
            options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            if (standalone && standalone->provider != "cpu" &&
                standalone->provider != "cuda")
                throw std::invalid_argument("provider must be cpu or cuda");
            if (standalone && standalone->provider == "cuda") {
                const auto available = Ort::GetAvailableProviders();
                if (std::find(available.begin(), available.end(),
                        "CUDAExecutionProvider") == available.end())
                    throw std::runtime_error("CUDAExecutionProvider is unavailable");
                const auto &api = Ort::GetApi();
                OrtCUDAProviderOptionsV2 *raw = nullptr;
                Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
                std::unique_ptr<OrtCUDAProviderOptionsV2,
                    decltype(api.ReleaseCUDAProviderOptions)>
                    cuda(raw, api.ReleaseCUDAProviderOptions);
                const char *keys[]{"device_id", "use_tf32"}, *values[]{"0", "0"};
                Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
                    cuda.get(), keys, values, 2));
                Ort::AllocatorWithDefaultOptions allocator;
                char *text = nullptr;
                Ort::ThrowOnError(api.GetCUDAProviderOptionsAsString(
                    cuda.get(), allocator, &text));
                standalone->cuda_options = text;
                allocator.Free(text);
                options.AppendExecutionProvider_CUDA_V2(*cuda);
            } else {
                Ort::ThrowOnError(
                    OrtSessionOptionsAppendExecutionProvider_CPU(options, 1));
            }
            if (standalone) {
                standalone->provider_registered = true;
                if (!standalone->profile_prefix.empty())
                    options.EnableProfiling(standalone->profile_prefix.c_str());
            }
            if (disable_intra_op_spinning)
                options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
            if (measurement) {
                measurement->spinning_explicit = disable_intra_op_spinning;
                measurement->spinning_entry_present =
                    options.HasConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning);
                if (measurement->spinning_entry_present)
                    measurement->spinning_entry =
                        options.GetConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning);
                if (disable_intra_op_spinning && measurement->spinning_entry != "0")
                    throw std::runtime_error("Spinning option readback mismatch");
                measurement->wall_begin = clock_ns();
                measurement->cpu_begin = process_cpu_ns();
            }
            session_ =
                std::make_unique<Ort::Session>(environment_, path.c_str(), options);
            if (standalone)
                standalone->session_created = true;
            if (measurement) {
                measurement->cpu_end = process_cpu_ns();
                measurement->wall_end = clock_ns();
                measurement->created = true;
            }
            if (std::string(Ort::GetVersionString()) != "1.20.1" ||
                session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
                throw std::runtime_error("Runtime/model contract mismatch");
            Ort::AllocatorWithDefaultOptions allocator;
            if (std::string(session_->GetInputNameAllocated(0, allocator).get()) !=
                    "input" ||
                std::string(session_->GetOutputNameAllocated(0, allocator).get()) !=
                    "output")
                throw std::runtime_error("Model names mismatch");
            // 导出模型仅 batch 为符号维；应用始终绑定 batch=1。
            const std::array<std::vector<int64_t>, 2> shapes{
                {{-1, 3, 224, 224}, {-1, 1000}}};
            for (size_t i = 0; i < 2; ++i) {
                auto type = i == 0 ? session_->GetInputTypeInfo(0)
                                   : session_->GetOutputTypeInfo(0);
                if (type.GetONNXType() != ONNX_TYPE_TENSOR)
                    throw std::runtime_error("Expected model tensor");
                const auto info = type.GetTensorTypeAndShapeInfo();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    info.GetShape() != shapes[i])
                    throw std::runtime_error("Model type/shape mismatch");
            }
            // 仅在 ORT setter、Session 创建和模型契约检查均成功后确认应用值。
            applied_intra_op_num_threads_ = intra_op_num_threads;
        }

        int intra_op_num_threads() const {
            return applied_intra_op_num_threads_;
        }

        bool session_alive() const {
            return session_ != nullptr;
        }

        std::string finish_profile() {
            Ort::AllocatorWithDefaultOptions allocator;
            return session_->EndProfilingAllocated(allocator).get();
        }

        void run(Tensor &input, Call &call) {
            const std::array<int64_t, 4> shape{1, 3, 224, 224};
            auto memory =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto tensor = Ort::Value::CreateTensor<float>(
                memory, input.data(), input.size(), shape.data(), shape.size());
            const char *in[]{"input"}, *out[]{"output"};
            call.begin = clock_ns();
            auto output =
                session_->Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);
            call.end = clock_ns();
            call.returned = true;
            if (output.size() != 1 || !output[0].IsTensor())
                throw std::runtime_error("Invalid runtime output");
            const auto info = output[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                info.GetShape() != std::vector<int64_t>{1, 1000})
                throw std::runtime_error("Runtime output contract mismatch");
            std::copy_n(
                output[0].GetTensorData<float>(), output_elements, call.logits.begin());
            call.output_available = true;
        }
    };
}
