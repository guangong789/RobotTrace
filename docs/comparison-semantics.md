# 运行比较语义（版本 1）

`comparison.compare_runs(baseline, candidate, allow_changes=(), metrics=...)` 是独立于 ROS、Docker、文件 I/O 和展示代码的纯函数。每个输入都是包含 `manifest`（run.json 数据）及 `analysis`（现有分析核心结果）的字典。它使用分析结果的统计量，不重新实现事件关联或分位数算法。

## 使用

先构建运行程序，再采集两次有界运行；CLI 会打印各自的运行目录：

```sh
./scripts/build.sh
python3 -m robottrace run --scenario slow --count 20 --rate 10 --work-ms 50
python3 -m robottrace run --scenario slow --count 20 --rate 10 --work-ms 10
python3 -m robottrace compare <baseline-dir> <candidate-dir> --allow-change processor_work_ms
python3 -m robottrace compare <baseline-dir> <candidate-dir> --allow-change processor_work_ms --json
python3 -m robottrace compare <baseline-dir> <candidate-dir> --allow-change processor_work_ms --output comparison.json
```

也可执行 `python3 tests/comparison_integration.py`，自动完成上述实验、公共 CLI 比较、未声明变量的拒绝比较，以及从持久化证据重新计算的一致性检查。它不会断言具体调度结果或改进比例。

默认比较 `processor_callback_ns` 和 `consumer_age_ns`。重复使用 `--metric <name>` 可选择现有八个指标。默认只打印摘要；`--json` 只输出 JSON；仅 `--output` 写文件，且不覆盖已有文件。输入目录不会被修改，也不更新其中的 analysis.json 缓存。

## 三层判断

结构化结果使用 `comparison_schema_version=1`。`conditions` 描述条件兼容性，`runs` 保留执行、采集、trace、观测、时钟及软件证据，各 `metrics` 独立给出统计差异、覆盖率、限制原因和判定。

- `COMPARABLE`：所选指标具备声明政策要求的证据，仅代表**已记录的条件**可比较。
- `NOT_COMPARABLE`：存在未解释配置变化、不兼容语义、无效 trace 或冲突时钟等明确障碍。
- `INSUFFICIENT_EVIDENCE`：缺少时钟/软件/预期集合等信息，或执行、采集、指标样本不完整。

顶层状态汇总所选指标；一个指标受限不会自动否定另一个证据充分的指标。优先级为 NOT_COMPARABLE、INSUFFICIENT_EVIDENCE、COMPARABLE，没有整体性能通过布尔值。

## 条件规则

接受当前版本 1 清单和结果、当前三进程管线布局、SingleThreadedExecutor 和 RELIABLE/VOLATILE/KEEP_LAST；不支持的管线或 QoS 语义即使两边相同，也不视为已证明可比较。normal/slow 是同一管线的参数预设。比较完整配置中的消息数、标称速率、QoS、drain、缓冲容量、域 ID、布局等；预期 source/sequence 集合按集合精确匹配，不只比较数量或首尾。

唯一支持的实验变量是 `processor_work_ms`，必须显式 `--allow-change processor_work_ms`，输出同时显示原值和新值。normal/slow 标签变化只有在确实存在已声明处理时长变化时才视为对应预设变化。其他字段或未知配置字段的差异不能绕过。缺少字段是证据不足，不等于已证明条件相同。

二进制 SHA256 和记录的 ROS 包版本必须一致；缺失则证据不足，已知变化则不可比较。当前不支持声明软件升级实验。比较程序没有假装清单记录了 Docker 镜像 ID、CPU、内核或负载；这些未控制因素限制了结论的强度。

每次运行必须有已验证的 CLOCK_MONOTONIC 域、零 monotonic/boottime offset、匹配的 host 和三节点时钟证据。不同运行可以有不同 boot UUID，不要求原始时间戳对齐。运行 UUID、PID、运行专属话题和命令中的这些标识不作为配置差异。

## 证据与指标

每个指标报告两边的 available/unavailable 数量、预期数量、预期身份覆盖数和覆盖比例、P50/P95，以及 candidate − baseline 的差值。百分比为差值 ÷ baseline × 100；基线为零、统计量缺失或结果不可表示时为 null，绝不以零补齐缺失延迟。

性能判定要求双方执行完成、采集完整、trace 有效、时钟/条件兼容，并且**该指标覆盖每一个相同的预期身份**。不设置任意覆盖率阈值，不把相同有效样本数误认为相同人群。不完整的全局 observations 不一定阻止独立指标：若该指标覆盖完整、采集与 trace 可靠，其他端点的缺失不会让它失效。派生 fixture 不支持真实运行性能判定。

判定仅按观察到的 P95：下降为 OBSERVED_IMPROVEMENT，上升为 OBSERVED_REGRESSION，相等为 NO_OBSERVED_DIFFERENCE；否则 WITHHELD。P50 仍单独报告，NO_OBSERVED_DIFFERENCE 仅指 P95。分析结果原有的阈值发现不参与相对判定；两边阈值保留供查看。

例如：

- 完整覆盖、匹配条件、已声明 50→10 ms 处理时长变化：可描述观察到的 P95 差异，不声称统计显著性或自动归因。
- candidate P95 较低但只覆盖 2/3 预期身份：保留描述性数值，WITHHELD。
- 同样的处理时长差异未声明、软件版本改变、未知配置漂移：NOT_COMPARABLE，WITHHELD。
- 缺少统计值、采集溢出或时钟验证：INSUFFICIENT_EVIDENCE；不能用实验声明豁免。

CLI 总是从原始事件重新调用现有分析器，避免陈旧 analysis.json。纯函数调用方也须提供相互对应的真实清单和分析结果；结构一致性检查不等于证据真实性认证。当前仍是有界数据集的内存分析，不是流式大轨迹平台。

## 退出码

- **0**：成功形成比较结果，包括 NOT_COMPARABLE 或 WITHHELD；不表示性能验收通过。
- **2**：参数、I/O、输入结构或版本不受支持，未形成有效比较，或指定输出文件已存在。

自动化调用应读取 JSON 中每个指标的 status、verdict 和 reasons；不要把进程退出码当作性能门禁。单对运行不提供假设检验、置信区间或统计显著性结论。
