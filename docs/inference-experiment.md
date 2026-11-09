# MobileNetV2 ROS2 CPU/CUDA 推理实验

当前唯一活跃推理模型为 MobileNetV2-12 FP32，默认使用 ONNX Runtime CPU 1.20.1。
CUDA Processor 需要显式选择独立 GPU 构建，Source/Consumer 继续使用 CPU 构建。
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
python3 -B experiments/cpu_inference/load_summary.py "<run-directory>"
python3 -B experiments/cpu_inference/latency_attribution.py "<run-directory>"
python3 -B experiments/cpu_inference/e3_summary.py "<e3-run-directory>"
python3 -B tests/inference_integration.py "<successful-vision-run-directory>"
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

## 显式 CUDA Processor

CPU 默认命令和 SDK 不变。GPU 复用 `pipeline.cpp`，以独立 CMake build tree、
`ROBOTTRACE_CUDA_BUILD=ON` 和 `/opt/rt/ort` GPU SDK 生成 `inference_pipeline_cuda`。
GPU 入口需要已经核验的依赖，构建脚本不下载依赖、不安装或重配系统环境：

- GPU ORT 1.20.1：`.deps/gpu/onnxruntime-linux-x64-gpu-1.20.1/`，与 CPU SDK 分开保存。
- cuDNN 9.3.0.75：`.deps/gpu/cudnn-linux-x86_64-9.3.0.75_cuda12-archive/lib/`。
- 既有 WSL GPU 驱动、`/usr/local/cuda-12.4/targets/x86_64-linux/lib/` 中所需运行库，
  以及已配置的 CDI 设备 `nvidia.com/gpu=all`。入口固定校验现有 Humble 镜像及 GPU ORT 库身份，
  仅挂载所需 CUDA/cuDNN 库；不修改 Docker/CDI 或全局加载配置。

为 CPU/GPU 分别选择尚不存在的 `bin/` 直接子目录和新日志。以下示例名称若已使用，应先换名。
CPU 的 `ROBOTTRACE_INFERENCE_OUTPUT` 覆盖拒绝旧 `bin/inference/`、已有目录和路径穿越；
省略覆盖时仍保持原 CPU 默认行为。本地历史 `bin/inference/` 不应作为下面的新构建目标。

```sh
ROBOTTRACE_INFERENCE_OUTPUT=bin/inference_cpu_checkpoint \
ROBOTTRACE_BUILD_LOG=evidence/checkpoint-cpu-build.log scripts/build-inference.sh
ROBOTTRACE_INFERENCE_OUTPUT=bin/inference_cuda_checkpoint \
ROBOTTRACE_BUILD_LOG=evidence/checkpoint-gpu-build.log scripts/build-inference-gpu.sh

python3 -B -m robottrace.inference --provider cuda \
  --cpu-artifact-dir bin/inference_cpu_checkpoint \
  --cuda-artifact-dir bin/inference_cuda_checkpoint \
  --count 5 --rate 5 --intra-op-threads 1
```

构建在隔离临时目录中执行，不申请 CDI 设备；GPU 构建中的 C++ 数值契约使用 CPU provider，
不能代替下面的实际 CUDA warm-up 验证。
运行器的 `--provider cuda --cuda-artifact-dir <GPU目录> --cpu-artifact-dir <CPU目录>`
分别校验两份清单，并仅用 GPU 二进制启动 Processor；Source/Consumer 使用 CPU 二进制和 CPU SDK。
已有 CPU 产物可配合 `--cpu-source-root <构建时实际源码快照>` 校验：原清单每项仍必须匹配，
仅源码从显式快照读取，二进制、SDK、模型和 fixture 仍读取实际文件。默认仍对当前源码校验。
快照必须在源码改动前保存，不能通过改写清单适配旧产物。

CPU 二进制接受省略 provider 或 `cpu`，拒绝 `cuda`。GPU 二进制只接受
`processor --provider cuda --intra-op-threads 1`，拒绝缺省、CPU、其他角色和四线程。
GPU 入口 `scripts/ros-inference-gpu.sh` 要求独立 `ROBOTTRACE_EVIDENCE_DIR`，
仅该目录可写；SDK、项目源码及固定 CUDA/cuDNN 库只读挂载。
运行显式申请 `nvidia.com/gpu=all`，保留原 host 网络、host IPC、domain 42 和 localhost/FastDDS 设置。
共享容器内所有角色均可见 GPU；这里保证的是 provider/构建/加载隔离，不是每进程设备隔离。
失败的新容器保留，成功后仅删除本次容器。

CUDA 初始化设置 device 0、`use_tf32=0`；请求、构建模式、注册、Session、readback、实际加载库路径/哈希分别留证。
CUDA Graph readback 缺失保持未知。五次原有数值 warm-up 后关闭 profile，按实际
`model_run` 区间和 PID/TID 唯一归组，每次必须为 101 个 CUDA 节点和
`Gather_100`、`Unsqueeze_101`、`Concat_102` 三个 CPU 辅助节点。
缺失、模糊归组、额外回退或加载库身份错误均阻止 ROS2 初始化，不用总节点数代替逐次证明。
`processor.warmup*.json`、`processor.cuda-execution.json` 和
`processor.warmup-numerical.jsonl` 保存在本次目录；warm-up 不进入九事件追踪，也不作为性能结果。

GPU 二进制的 `--warmup-only` 在上述门禁后、`rclcpp::init` 前退出，供隔离初始化验证。
它不创建 ROS2 节点或处理 Source 消息，不能证明三进程功能、readiness 或性能。
同步 `Session::Run()`、主机 tensor、输出复制、数值容差、QoS、所有 readiness 预算及消息计时边界保持原样。

## GPU 接入的验证边界

以下是本地保留的历史结果，证据目录不随源码发布：

| 阶段 | 已建立的结果 | 本地证据目录 |
| --- | --- | --- |
| G1-02B0 | 独立 CPU 重建和五消息 smoke 通过，45 个完整事件。 | `evidence/g102b0_20261109T070857Z/` |
| G1-02B1/B1R | 独立 CPU/GPU 构建完成；B1R 实际 CUDA 门禁通过，五次各 101 CUDA 节点和三个已知 CPU 辅助节点，5,000 个 warm-up 输出通过独立参考校验。 | `evidence/g102b1_20261109T073526Z/`、`evidence/g102b1r_20261109T080410Z/` |
| G1-02C、C-R1 | 两次未添加图诊断的 GPU 功能尝试均为 FAIL：冻结 `1/1/0/1`，未释放 Source。 | `evidence/g102c_20261109T114310Z/`、`evidence/g102c_r1_20261109T120407Z/` |
| G1-02E-R2 | 添加临时图诊断的运行达到 `1/1/1/1`，五消息功能与数值校验通过，45 个事件、零丢弃；诊断后处理失败，原任务仍为 PARTIAL。 | `evidence/g102e_r2_20261109T125838Z/` |
| G1-02E-R3 | 纯离线恢复全部 51 条图观测；采样的清理前区间存在 Processor Reader 图可见性缺失，根因仍未确认。未改写 R2 的 PARTIAL。 | `evidence/g102e_r3_20261109T132252Z/` |

临时端点图诊断及其恢复工具只在本地 evidence 中，未进入生产 pipeline。
目前没有未添加图诊断的 GPU 三进程功能 PASS；上述 instrumented smoke 不证明 DDS 启动可靠性已解决。
图可见性、matched-count、ready 持久化和实际消息交付是不同观测，跨进程采样并不同时发生。
这个工程检查点不宣称 ROS2 GPU 加速比、吞吐或延迟改进。
