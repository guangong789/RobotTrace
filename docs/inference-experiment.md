# MobileNetV2 ROS2 CPU 推理实验

当前唯一活跃推理模型为 MobileNetV2-12 FP32，使用 ONNX Runtime CPU 1.20.1。
复用原 Humble 镜像与 V1 事件采集、分析语义；不改变 V1 workload、上游集成或 Comparison Core。

## 准备、构建和运行

在仓库根目录执行：

```sh
./scripts/setup-inference.sh
# 首次使用：在临时 Python 环境中安装模型说明列出的固定依赖，再生成 fixture。
python3 -B models/mobilenet/prepare.py
./scripts/build-inference.sh
python3 -B -m robottrace.inference --count 5 --rate 5
python3 -B -m robottrace.inference --count 150 --rate 30 --intra-op-threads 4
python3 -B -m robottrace.inference --startup-only
```

已有正确 fixture 时无需重新生成。模型与照片的固定 revision、SHA-256、预处理及独立参考来源见
[模型说明](../models/mobilenet/README.md)；无 ROS2 的验证见
[独立程序](../experiments/cpu_inference/README.md)。生成参考需要 ONNX 1.17.0、NumPy 1.26.4、Pillow 9.0.1，
runner 仅需 Python 标准库。下载 SDK 与模型不安装系统软件，也不复制或重建镜像。
ROS2 构建/运行需要 Linux amd64、Python 3.10+、Docker 及按根 README 准备的 Humble 镜像；
首次 fixture 准备另需带 pip 的 Python 3.10 环境，完整命令见模型说明。全部 Python 单元测试也需要这些 fixture。

构建在原 Docker 环境中使用 GCC 11.4 / C++17，在容器临时目录重新生成接口和程序、运行 C++ contract，
仅将运行库、可执行程序和构建身份保存至 ignored `bin/inference/`。
源文件、fixture、模型、SDK 库及运行产物受构建记录/哈希校验约束；过期产物必须重建。
设置 `ROBOTTRACE_BUILD_LOG` 可将新构建日志放入独立目录，保留旧日志。

## 数据与执行契约

三个独立 C++ 进程按 Consumer→Processor→暂停的 Source 启动，均使用 SingleThreadedExecutor。
输入为实际 FP32 `[1,3,224,224]` tensor，输出为 `[1,1000]` logits，不应用 softmax。
消息保留完整 UUID、source token、sequence、生成时间和 sample_id。
样本为同一真实照片预处理后的中心裁剪及水平翻转，并非两张独立照片；sample_id 不替代完整关联身份。
Processor 核对接收的 tensor，Consumer 独立检查身份、重复、有限值及全部 1000 个参考输出。
参考由 ONNX NumPy ReferenceEvaluator 生成，不是通过被测 ORT 生成；容差为 atol=rtol=1e-4。

Processor 只初始化一个 Session，交替输入完成五次 warm-up 后才标记初始化成功。
CPUExecutionProvider 固定 sequential、inter-op 1（sequential 下不参与并行执行）、basic 图优化及 CPU arena。
`--intra-op-threads` 仅支持 1/4，默认保持 1，整个运行使用同一 Session 配置。
Processor 在 ORT setter、Session 创建和模型契约检查成功后记录应用值；runner 核对启动、ready 和终态证据。
这证明配置已成功交给 Session，不等于测量到每次推理都有对应数量的 OS 线程同时运行。
`--count` 默认 20，范围 1–1000；`--rate` 默认 5 Hz，允许有限的 1–10000 Hz，计划首末发布窗口须不超过十秒。
Source 保持绝对目标时间，迟到时追赶，不跳过 sequence；publish 调用可能阻塞，目标频率不等于实测频率。
V1 Comparison Core 对推理工作负载的可比性策略保持不变。

## 严格 readiness 与失败证据

QoS 固定为 RELIABLE / VOLATILE / KEEP_LAST(10)。Source 放行前必须满足四个实际匹配条件：
Source 的 Subscriber、Processor 的 Publisher/Subscriber、Consumer 的 Publisher 计数都恰为 1，
同时验证实际 QoS、初始化和五次 warm-up。不接受零匹配覆盖、额外匹配或以业务活动代替 readiness。
每个进程初始化等待仍为 25 秒，完整匹配等待仍为 15 秒，发布与消费完成等待仍为 15 秒；节点有 60 秒安全边界。
启动标记核对本次 run_id 与实际子进程 PID。清理复用既有有界 SIGINT/kill 行为。

基础状态通过公共 ROS2 matched-count 和 actual-QoS API 提供。启动阶段每 250ms 或状态变化时更新状态文件，
runner 在原轮询中保留首条、最新观测及最多 64 条变化记录；超过上限保留首条和最近 63 条，明确报告截断及丢弃数。
读取错误保持显式，不转换成零计数。观测时间是轮询/公共 API 采样时间，不是 DDS endpoint 精确创建时间。

每个新运行使用独立 `evidence/vision_cpu_<UUID>/`，startup-only 使用 `vision_startup_<UUID>/`。
失败的 `run.json` 保留 `execution_error`、`pre_cleanup_readiness` 和
`readiness_diagnostics.pre_cleanup`；后者在子进程 shutdown 前冻结，不会被终态 ready.json 覆盖。
终态文件不足以解释瞬态启动失败，应先看冻结快照、未满足条件、历史和读错误。
诊断失败不会跳过子进程清理。缺失 collector 状态保持 unknown，不能报告为零丢弃。
`--startup-only` 永不释放 Source，不宣称数值正确性或推理性能已验证。

WSL2/Docker 原默认 localhost 环境曾出现间歇 discovery/readiness 超时，根因仍未证实。
正常运行保留 `ROS_LOCALHOST_ONLY=1`、原 UDP+SHM 环境和原 ROS2 wrapper。
历史 R01–R05 记录仍在 evidence；原生 Writer、详细 graph/GID、UDP-only 和反转启动选项已退出活跃代码。
原始失败不改写，超时不自动重试至成功，不延长预算，不把启动失败归为模型性能失败。
调查索引可见 `evidence/r02_20261106T124533Z_b9c7dd/` 至
`evidence/r05_20261106T233119Z_c639d1/` 对应目录。

## 计时与离线检查

每个正常消息仍有原九事件。每进程 collector 容量为 4096，记录实际丢弃；warm-up 不产生测量消息。
`run.json` 保存模型/fixture/构建身份、运行镜像、独立 host/container/node 单调时钟验证及实际活动计数。
原始事件、节点/collector 状态和有界 Processor/Consumer 全量结果支持独立复核。

- 同步 inference API duration：仅包围 `Session::Run`；包含 ORT 分派、分配及调度影响，不等于纯模型计算。
- Processor callback：包含接收后的 tensor 处理、API 调用、校验和发布等回调工作，warm-up 排除。
- Consumer age：Source generate 到 Consumer callback entry 的跨进程区间；须有可信时钟和完整身份。
- 延迟归因按 `(run_id, source_id, sequence)` 配对：A=generate→Processor entry，
  B=Processor entry→output publish begin，C=output publish begin→Consumer entry，逐消息满足 A+B+C=age。
  A/C 不能单独解释为纯 DDS 或纯 Executor 排队；不同区间的 P95 不能直接相加。

只读离线摘要命令：

```sh
python3 -B experiments/cpu_inference/load_summary.py <run-directory>
python3 -B experiments/cpu_inference/latency_attribution.py <run-directory>
python3 -B experiments/cpu_inference/e3_summary.py <e3-run-directory>
python3 -B tests/inference_integration.py <successful-vision-run-directory>
python3 -B -m unittest discover -s tests
```

load summary 区分目标频率、实际发布/完成 cadence 和各自时间窗口；延迟摘要保留不可用/缺失身份。
例如历史 `vision_cpu_4e208a170b6542f29f94d88a64bd9acb` 仅有 378/450 条完整结果，
可用样本可能存在选择偏差，不能当作完整运行，也不能用其分位数解释所有消息。
未通过时钟/完整性条件时相关区间保持不可用。V1 Comparison Core 对推理工作负载仍为 NOT_COMPARABLE，
不得将离线描述性测量解释为已有性能改进判定。

E3 摘要分别保留请求/应用配置、冻结匹配状态、启动失败、实测发布率和完成 cadence，
并对实际存在的全量结果再次独立校验。没有发布消息的启动失败不计作零推理吞吐；
部分结果的覆盖率以成功发布数为分母，A/B/C 分布仅代表具备完整关联端点的身份。
Processor 的可选 CPU 时间/峰值 RSS 覆盖初始化、warm-up、等待、验证和输出等进程生命周期，不能直接解释为单次推理成本。

历史证据不重写、不自动压缩；SDK、模型数据和构建产物与源码分开保存。
文中的历史 evidence 路径指本地保留记录，不随 Git 源码发布；新检出通过新的运行生成自己的证据。
