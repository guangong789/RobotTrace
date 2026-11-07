# E4 standalone CPU 资源归因

使用原 Humble 容器、MobileNetV2-12 FP32、ORT CPU 1.20.1 和原两份 fixture。
保持 sequential、inter-op 1、basic 图优化、CPU arena 和数值容差不变。

```sh
vision_inference models/mobilenet --intra-op-num-threads 4 --resource-phases
# 仅在预声明条件满足后，四线程对照可指定：
vision_inference models/mobilenet --intra-op-num-threads 4 --resource-phases --intra-op-spinning disabled
python3 -B experiments/cpu_inference/test_resource_phases.py BINARY models/mobilenet
```

资源模式固定顺序：Session 构造 → 五次 warm-up → 一百次真实推理 → 同一 Session 存活时空闲约两秒。
空闲期只有有界 CLOCK_MONOTONIC sleep 和边界时钟读取，不执行 Run、日志或 tensor 校验。
CLI 在资源模式只接受 1/4 线程，禁用 spinning 只接受 4 线程。普通 standalone 仍支持原 1/2/4。

wall 使用 CLOCK_MONOTONIC，CPU 使用 CLOCK_PROCESS_CPUTIME_ID（整个进程全部线程），并记录各自分辨率。
每个区间先读 wall/CPU 起点，结束时读 CPU/wall 终点，CPU 采样区间位于 wall 区间内部。
分辨率不等于准确度；计时调用与非同时采样有小额边界开销。CPU/wall 可以超过 1，并非某个 worker/core 的利用率或功耗。

Session 区间准确包围共享模型实现中的 `make_unique<Ort::Session>`；ORT 内部的模型文件读取、
图初始化在此边界内。Ort::Env、SessionOptions 和创建后的模型形状/名称检查不在该区间内，
所以四个阶段之和不是全生命周期成本。fixture 文件加载单独记录在 Session 之前。
warm-up/active 阶段包含输入包装、输出复制、1000-output 参考校验及观测存储，
这些均在单次同步 Run 的 API 计时之外；阶段 CPU 不是纯模型计算成本。

SDK 1.20.1 的 `onnxruntime_session_options_config_keys.h` 及
[对应版本的线程池初始化](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/session/inference_session.cc#L450-L463)
确认 `session.intra_op.allow_spinning` 缺省按 `1` 处理。默认不添加这个 entry；禁用时仅添加 `0`，
通过 SessionOptions 的 HasConfigEntry/GetConfigEntry 回读并记录，再创建 Session。
这不是对每个 worker 实际调度策略的运行时 getter；实验不改变 inter-op 或 force-spinning-stop。

`resource_cost.py BINARY EVIDENCE_DIR` 读取该目录预声明的 plan.json，在同一容器中依次执行
A1→B1→B2→A2，A=1、B=4，全部使用默认 spinning。条件满足后才执行
B-default→B-no-spin→B-no-spin→B-default；不重试、不新增配置。
门槛为两次 B 的 idle CPU 均至少 20ms、均至少为最大 A 的十倍，且均高出最大 A 至少 10ms。
这是探索性 materiality 门槛，不是显著性检验。记录失败、原始 API 样本和所有阶段边界，不移除慢样本。

空闲测量紧随 active 阶段，可能包含有限的 post-Run spinning burst；不能据两秒积分认定长期持续空转。
本实验不能把历史 ROS2 Processor 整进程 CPU 总量重新分解成这些阶段，也不改变 ROS2 默认设置。
