// 直接测试生产环境使用的 C++ 身份提取器，不复制其实现。
#include "identity.hpp"
#include <iostream>

int main() {
    unsigned checks = 0;
    auto require = [&](bool condition) {
        ++checks;
        if (!condition)
            throw std::runtime_error("Identity contract failed");
    };
    require(rt_external::sequence("Hello World: 1", 20) == 1);
    require(rt_external::sequence("Hello World: 20", 20) == 20);
    for (const auto *bad :
        {"", "Hello, world! 1", "Hello World: ", "Hello World: 0", "Hello World: 01",
            "Hello World: +1", "Hello World: -1", "Hello World: 1 ", "Hello World: 1x",
            "Hello World: 21", "Hello World: 18446744073709551616"}) {
        bool rejected = false;
        try {
            rt_external::sequence(bad, 20);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected);
    }
    rt_external::Identities ids(20);
    require(ids.accept("Hello World: 4") == 4); // 缺失前缀不改变来源身份。
    bool duplicate = false;
    try {
        ids.accept("Hello World: 4");
    } catch (const std::invalid_argument &) {
        duplicate = true;
    }
    require(duplicate);
    std::cout << "PASS identity contract: " << checks << " checks\n";
}
