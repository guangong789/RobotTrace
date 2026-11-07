// 实际测试生成消息的序列化、身份、完整载荷、照片模型输出及拒绝条件。
#include "inference.hpp"
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <iostream>
#include <limits>
#include <type_traits>

template <class Message> Message roundtrip(const Message &original, size_t &size) {
    rclcpp::Serialization<Message> serialization;
    rclcpp::SerializedMessage bytes;
    serialization.serialize_message(&original, &bytes);
    size = bytes.size();
    Message result;
    serialization.deserialize_message(&bytes, &result);
    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected fixture path");
        using namespace inference;
        static_assert(std::tuple_size<decltype(Input{}.input)>::value == 150528);
        static_assert(std::tuple_size<decltype(Result{}.logits)>::value == 1000);
        static_assert(std::is_same<decltype(Input{}.input)::value_type, float>::value);
        static_assert(
            std::is_same<decltype(Result{}.logits)::value_type, float>::value);
        unsigned checks = 0;
        auto require = [&](bool value) {
            ++checks;
            if (!value)
                throw std::runtime_error("Inference contract failed");
        };
        Fixtures fixtures(argv[1]);
        CpuModel model(std::string(argv[1]) + "/data/mobilenetv2-12.onnx");
        require(model.intra_op_num_threads() == 1);
        require(processor_threads("1") == 1 && processor_threads("4") == 4);
        for (const std::string value : {"0", "2", "-1", "4x", "", "1.0"}) {
            bool rejected = false;
            try {
                processor_threads(value);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            require(rejected);
        }
        CpuModel candidate(std::string(argv[1]) + "/data/mobilenetv2-12.onnx", 4);
        require(candidate.intra_op_num_threads() == 4);
        for (uint32_t warmup = 0; warmup < 5; ++warmup) {
            for (auto instance : {&model, &candidate}) {
                Call call;
                instance->run(fixtures.input[warmup % sample_count], call);
                require(call.returned && fixtures.correct(call.logits, warmup % sample_count));
            }
        }
        Identity identity("0000000000000001ffffffffffffffff", 20);
        Results results;
        size_t input_bytes = 0, output_bytes = 0;
        for (uint64_t seq = 1; seq <= 20; ++seq) {
            Input input;
            identity.fill(input, seq);
            input.generated_ns = 100;
            input.input = fixtures.input[input.sample_id];
            auto transported = roundtrip(input, input_bytes);
            require(identity.valid(transported, 101));
            require(fixtures.payload(transported));
            require(transported.generated_ns == input.generated_ns);
            Call call;
            model.run(transported.input, call);
            require(call.returned && call.output_available && call.end >= call.begin);
            require(fixtures.correct(call.logits, input.sample_id));
            Call accelerated;
            candidate.run(transported.input, accelerated);
            require(accelerated.returned && accelerated.output_available);
            require(fixtures.correct(accelerated.logits, input.sample_id));
            Result result;
            identity.fill(result, seq);
            result.generated_ns = input.generated_ns;
            result.logits = call.logits;
            const auto received = roundtrip(result, output_bytes);
            require(received.logits == result.logits);
            require(results.accept(received, 101, identity, fixtures));
        }
        require(input_bytes == 602160 && output_bytes == 4048);
        Result result;
        identity.fill(result, 1);
        result.generated_ns = 100;
        result.logits = fixtures.expected[0];
        require(!results.accept(result, 101, identity, fixtures) &&
            results.duplicates == 1);
        Results negative;
        auto invalid = result;
        invalid.sample_id = 1;
        require(!negative.accept(invalid, 101, identity, fixtures));
        invalid = result;
        invalid.run_lo ^= 1;
        require(!negative.accept(invalid, 101, identity, fixtures));
        invalid = result;
        invalid.source_id = 2;
        require(!negative.accept(invalid, 101, identity, fixtures));
        invalid = result;
        invalid.generated_ns = 102;
        require(!negative.accept(invalid, 101, identity, fixtures));
        invalid = result;
        invalid.logits[0] = std::numeric_limits<float>::quiet_NaN();
        require(!negative.accept(invalid, 101, identity, fixtures));
        invalid = result;
        invalid.logits[0] += 10;
        require(!negative.accept(invalid, 101, identity, fixtures));
        Input bad;
        identity.fill(bad, 1);
        bad.generated_ns = 100;
        bad.input = fixtures.input[0];
        bad.input[0] += 1;
        require(!fixtures.payload(bad));
        require(results.valid == 20 && negative.valid == 0 && negative.invalid == 6);
        const Schedule schedule(1000, 1000);
        require(schedule.period_ns == 1000000ULL);
        require(schedule.target(123, 1) == 123);
        require(schedule.target(123, 1000) == 999000123ULL);
        // 目标不依赖当前迟到量，不做相对睡眠或跳号。
        require(schedule.target(123, 3) == 2000123ULL);
        Identity large("0000000000000001ffffffffffffffff", 1000);
        Results boundary;
        large.fill(result, 1000);
        result.generated_ns = 100;
        result.logits = fixtures.expected[999 % sample_count];
        require(boundary.accept(result, 101, large, fixtures));
        result.sequence = 1001;
        require(!boundary.accept(result, 101, large, fixtures));
        for (double rate :
            {0.0, std::numeric_limits<double>::infinity(), 10001.0, 5.0}) {
            bool rejected = false;
            try {
                Schedule invalid_schedule(1000, rate);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            require(rejected);
        }
        std::cout << "PASS inference contract: " << checks
                  << " checks; two photo tensors; 1/4-thread sessions; five warm-ups each; "
                     "40 measured CPU calls; input CDR bytes="
                  << input_bytes << "; output CDR bytes=" << output_bytes << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
