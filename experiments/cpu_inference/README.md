# MobileNetV2 独立 CPU 验证

`vision_inference` 使用与 ROS2 pipeline 相同的 `mobilenet.hpp`，在无 ROS2 节点的情况下验证固定照片输入和全部 1000 个输出。
唯一活跃模型为 MobileNetV2-12 FP32；SDK 为官方 ONNX Runtime 1.20.1 Linux x64 CPU 版本。

在仓库根目录执行：

```sh
./scripts/setup-inference.sh
# 首次使用需要生成 fixture，依赖和来源见 models/mobilenet/README.md。
python3 -B models/mobilenet/prepare.py
cmake -S experiments/cpu_inference -B build/vision_inference -DCMAKE_BUILD_TYPE=Release
cmake --build build/vision_inference --parallel 2
ctest --test-dir build/vision_inference --output-on-failure
```

已有通过哈希检查的 fixture 时无需重新生成。构建需要 Linux x86_64、C++17 编译器和 CMake 3.16+；
获取资源还需要 curl、tar 和 sha256sum。运行不需要 Python 推理包，不安装系统软件，不重建 ROS2 镜像。
`.deps/`、模型二进制与生成构建目录保持在 Git 之外。

SDK 下载地址及 SHA-256 固定在 `scripts/setup-inference.sh`；SDK 自带许可证和第三方声明保留在下载目录中。
模型、照片、预处理、独立 NumPy 参考及 tensor 哈希见 [模型说明](../../models/mobilenet/README.md)。

每个进程创建一个 CPU Session，固定 sequential、inter-op 1、basic 图优化及 CPU arena。
`--intra-op-num-threads N` 显式调用 ORT session API，支持本轮固定候选 1、2、4，默认仍为 1。
共享模型构造函数的默认值保持 1；standalone 不启动 ROS2 pipeline。
交替使用两份输入，先执行五次 warm-up，再执行一百次测量；每次检查全部输出，容差为
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，不只比较 top-1。
这验证固定输入的数值正确性，不构成数据集准确率评估。

API duration 使用 CLOCK_MONOTONIC 包围同步 `Session::Run`，排除模型加载、warm-up 和输出校验；
包括运行时分派、输出分配及宿主调度影响，并非纯计算时间。程序同时报告本进程峰值 RSS 和 CPU 时间。
独立程序不验证 ROS2 消息、跨进程时钟或端到端延迟；这些由 [ROS2 推理实验](../../docs/inference-experiment.md) 验证。

单进程示例：

```sh
build/vision_inference/vision_inference models/mobilenet --intra-op-num-threads 2
```

stdout 为 JSON，包含精确配置、warm-up 边界和每次测量的 iteration、sample_id、单调时钟区间、
duration 及完整输出校验结果。P50/P95 使用最近秩法；所有原始有效样本保留，不删除慢样本。
失败返回非零状态、stderr 原因和带 `failed` 状态的已收集观测；缺失调用不补零。

`characterize.py BINARY EVIDENCE_DIR` 在同一环境中顺序执行六个独立进程，固定顺序为
1→2→4→4→2→1，不重试、不搜索其他配置。证据目录须事先创建；已有实验结果时拒绝覆盖。
该脚本要求 `ROBOTTRACE_IMAGE_ID` 记录实际容器镜像 ID，保存源码/二进制/模型/fixture 哈希、
CPU/编译器/运行环境、进程退出状态、原始 JSON 和 stderr。实验应在已验证的原 Humble 容器中运行，
不能混合 host 与 container 结果。CTest 中的配置契约属于验证，不属于六次正式测量。

独立程序与 ROS2 使用共享模型构造入口，默认设置均为 1。共享头文件哈希变化后，
运行 ROS2 前须遵守原有构建溯源校验并重建；standalone 的构建和测试独立进行。
