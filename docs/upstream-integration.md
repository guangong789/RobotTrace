# 独立 ROS2 应用接入：限定的 Talker → Relay → Listener

这是经过打点后重新构建的上游应用，不是对已安装原版二进制的无侵入跟踪。
只支持一个发布源、`std_msgs/msg/String` 和三个独立的单线程进程；不扩展分析核心、事件 schema 或 R1-02 比较政策。

## 来源和本地改动

`ros2/robottrace_upstream/upstream.json` 固定仓库、完整提交、每个原始文件的 URL 与 SHA256：

- `ros2/demos`：`1fc7a62ee591e4c4a7f84e72aa4b33de11f6fbf3`。
- `ros-tooling/topic_tools`：`0fc927b7c0af0aaffae34a947b6ea4a7f9f97c94`。

`upstream/` 是未修改的选定源码及两份 Apache-2.0 许可证，保留版权声明。
`instrumentation.patch` 在临时构建目录修改四个上游文件，并在修改处标明 RobotTrace 增补。
不下载、安装或构建整个 demos/topic_tools 仓库；本地 CMake 只构建选中的 Talker、Listener、Relay 及其直接实现。
所需依赖为现有 Humble 的 ament_cmake、rclcpp、rclcpp_components、std_msgs，以及 C++17 编译器、CMake、patch。
这些 Relay 源文件不引用 topic_tools_interfaces，因此这个限定构建不需要安装该包。

上游行为保留为：Talker 每秒生成 `Hello World: N` 并发布，Listener 在回调中记录日志，Relay 使用 GenericSubscription/GenericPublisher。
Relay 仍在原来的互斥锁内执行 `pub_->publish(*msg)`，输出的是原始 SerializedMessage。
新增反序列化仅用于读取身份，不重新编码输出；对实验之外的无效身份输入，适配器拒绝记录和处理，并报告运行失败。

新增 `experimental_control.hpp` 只负责文件确认的暂停/恢复、20 条上限和发布阶段确认。
未获允许的 timer tick 不生成消息，也不推进计数；保留上游 1 秒 timer，不添加模拟工作或替代 relay。
trace.hpp 复用原有 Buffer/Clock，负责有限身份检查、九事件记录、端点快照和退出刷新。
Listener 增加 100 ms 端点观察 timer，Relay 复用原有发现 timer，Talker 在其 timer 中观察；快照只在状态改变时写出。

## 构建和执行

```sh
./scripts/build-upstream.sh
python3 -B -m robottrace.upstream normal
python3 -B -m robottrace.upstream late-relay
python3 -B tests/upstream_integration.py
```

构建输出为 `bin/upstream/` 的三个可执行文件、两个应用库及 build.json。
临时构建/安装树随容器销毁。build.json 记录原始来源、补丁和头文件等构建输入、各可执行文件、应用库、ldd 解析库、动态 RMW/Fast DDS 插件及包版本。
每次实验校验这些身份并写入 run.json；另从三个真实进程的 /proc/PID/maps 记录已加载 ROS/应用库及哈希，补足动态类型支持库。不使用旧 pipeline 哈希代表三份程序。
`ROBOTTRACE_BUILD_LOG`、`ROBOTTRACE_EVIDENCE_DIR` 与已有脚本一致，可指定本次验证的独立输出目录。
重新打点或修改构建输入后必须重新构建；无需重新构建原有 ROS 镜像或 pipeline。

## 身份与时间

严格接受 `Hello World: ` 后接规范的、无前导零十进制整数，范围 1..20；拒绝额外字符、溢出和重复值。
三个进程共享配置的 run_id 和固定 source_id `demo:talker`。
String 中只有序列，没有 UUID 或源时间戳。run.json 明确记录 `run_uuid_in_payload=false`。
每次创建全新 UUID 话题命名空间，限定一个声明发布源；运行身份来自这种显式实验隔离，不是载荷认证或跨运行重放检测。
源端生成时刻来自源端真实事件记录，不从接收次序、日志文本时间或 ROS Header 时间推断。

回调入口先采 CLOCK_MONOTONIC，再做身份解析，最后用 record_at 记录入口。
Relay 回调时间包含反序列化身份读取、锁、记录和原始发布；Listener 回调时间包含身份检查及原有日志调用。
消费入口可能早于 Relay 发布返回或回调退出，这是合法重叠。
发布区间包含少量打点成本；未量化完整接入开销，不宣称零开销或纯 DDS 时延。
时钟沿用宿主/容器和三个节点的 boot ID、零 time-namespace offsets 校验；不支持多主机同步。

## 场景和状态

正常场景启动三进程，确认四个端点的实际匹配数及实际 QoS，并保存 ros2 topic info --verbose 的图观察，然后允许第一次发布。
Talker 深度为上游的 7；Relay 输入、输出及 Listener 深度为 10；均为 RELIABLE、VOLATILE、KEEP_LAST。
实际 QoS 以节点 get_actual_qos 快照为准，图观察输出原样保留，未知字段不改写成猜测值。

晚 Relay 场景先仅启动 Listener 和 Talker，确认源没有匹配订阅者，允许序列 1..3。
源在第三次 publish 成功返回后写 prefix.json，保持下一条暂停。
运行器确认该文件和零匹配数，再启动 Relay；双链路匹配与 QoS 观察完成后才恢复序列 4..20。
阶段推进靠实际确认和状态谓词；20 ms 轮询只是等待确认，没有用固定睡眠代替阶段条件。
末条消费回调完成后停止，所有等待有超时，不用假定的 drain 睡眠判断完成。

正常预期 180 个事件、20 个完整身份。晚 Relay 预期 162 个事件、17 个完整身份；源仍有全部 20 个身份。
后者应为 Collection Complete、Observations Incomplete，缺少 18 项下游观测。
前缀消息的下游七个指标为 null/MISSING_ENDPOINT；源发布区间仍可用。
缺失原因需要结合已确认的实验阶段与 VOLATILE 语义解释，单看缺失事件不能断言网络丢包。

## 故障与验证边界

历史实验中曾在正常场景和晚 Relay 场景出现间歇 readiness 超时。
具备冻结诊断的一次失败显示 Relay 输出端点存在，但报告的订阅匹配数为 0、要求为 1；这仍不足以确认 DDS 根因。
当前保留该已知限制，不改变精确匹配条件、超时或 QoS 来宣称修复。

运行器复用 H01 的逐进程有限 SIGINT/强杀清理；独立解析状态、保留主错误、尽力保存失败清单。
失败发生时先保存端点状态快照，避免关闭过程改变匹配数后丢失诊断信息。
`run.json.readiness_diagnostics` 记录现有匹配轮询的单调观察时间、文件存在性、可知的端点存在性、报告计数、未知/坏数据及未满足条件；不增加 DDS 探针，也不改变放行条件。
只保留状态变化，`poll_count` 仍计入未变化轮询。`entries` 最多 32 条，保留首条和最近条目（包括成功/超时边界）；超限淘汰中间旧记录，并显式设置 `truncated`、累计 `dropped_entries`。另固定保留最新实际轮询 `last_observation` 和失败时的清理前快照 `pre_cleanup` 各一份。
边界记录的 `boundary_ns` 与最后一次实测的 `observed_ns` 分开，`state_reobserved=false` 表示边界没有重新读取文件。观察时间是运行器读文件的时间，不是节点采样时间或 DDS 端点创建/匹配变化的准确时间；文件也可能保留旧报告，逐文件读取不是跨节点原子快照，两次轮询之间的变化可能未被观察到。
失败时查看 `readiness_diagnostics.pre_cleanup`、`last_observation`、终止边界和原有 `failure_readiness`；它们在清理前冻结，`frozen_before_cleanup=true`。终态 ready.json 可能已被关闭更新覆盖，不能代替冻结快照。成功时保留 readiness 阶段的成功观测，不额外采集关闭前端点。
诊断仅在内存中收集，随原清单在子进程清理后保存；诊断错误保留总数和最多 4 条限长错误样本，存在错误或截断时不能把历史当作完整。清单写失败仍显式报告主错误，不承诺存储故障下保留诊断。
状态未知不填零 collector；未刷新不能标为正常完成。清单写失败会显式保留原始错误上下文。
进程崩溃、SIGKILL、断电或存储故障不保证缓冲证据持久化。

构建会直接运行 C++ 身份契约测试；Python 单元测试覆盖阶段顺序、主错误、退出超时、清单失败和来源身份。
真实接入测试验证正常/晚启动结果和原始证据重分析一致性，不断言精确调度时延。
run.json 使用明确的 upstream 场景及实际配置，现有 compare 的拒绝/受限结果不代表已支持外部应用比较。
本次接入只证明兼容当前一对一三阶段契约的独立应用可以复用核心，不推广到任意拓扑、多个发布源或并行 Buffer。
