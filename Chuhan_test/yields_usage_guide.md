# Quokka 使用 slug2 产量表指南

## 1. 你现在手里的表有哪些

目录在 yields，主要包含：

1. SNII_Sukhbold16
2. SNII_Kobayashi0611
3. AGB_Karakas16
4. superAGB_Doherty14
5. isotope_data.txt

建议优先组合：

- SNII: Sukhbold16
- AGB + superAGB: Karakas16 + Doherty14

这是 slug2 当前默认支持的组合思路（对应其 yield_mode）。

## 2. 每类文件的格式与读取要点

## 2.1 SNII_Sukhbold16

文件名模式：

- s20.0.yield_table
- s25.3.yield_table
- ...

其中质量来自文件名（20.0, 25.3 等，单位 Msun）。

文件内容有两种头：

1. [isotope] [wind]
2. [isotope] [sn] [wind]

含义：

- wind: 前超新星恒星风贡献
- sn: 爆发抛射贡献

行格式：

- c12  1.23E-02  4.56E-03
- 或 c12  1.23E-02

实现建议：

1. 用文件名提取初始质量 M_init。
2. 每个 isotope 存两套表：yield_wind(M), yield_sn(M)。
3. 质量维度用 Steffen/Akima 插值都可，避免高阶震荡。
4. 若某文件仅有 wind 列，sn 列置 0。

## 2.2 SNII_Kobayashi0611

目录结构按金属度分子目录：

- z001models
- z004models
- z02models

每个目录内是若干质量点文件：

- s13.0.yield_table, s15.0.yield_table, ...

文件头示例：

- [isotope] [ejecta]

行格式：

- c12  9.74E-02

实现建议：

1. 从目录名提取 Z（如 z004 对应 Z=0.004）。
2. 从文件名提取质量 M_init。
3. 构建二维插值表 Y(M, Z)。
4. 该表是总抛射，不分 wind/sn 列。

## 2.3 AGB_Karakas16

关键文件在：

- AGB_Karakas16/z014models/*.dat
- AGB_Karakas16/z007models/*.dat
- AGB_Karakas16/z03models/*.dat

单文件头示例列：

- species, A, yield, mass(i)_lost, mass(i)_0, ...

重要点：

1. 第 3 列 yield 是净产额（net yield）。
2. 第 4 列 mass(i)_lost 是抛射总量（gross ejecta of isotope）。

若你希望与 slug2 读取行为一致：

- 用 mass(i)_lost（第 4 列）作为注入量。

若你要做“净产额模型”：

- 用 yield（第 3 列）。

此外，文件名里还编码了 PMZ 和 overshoot 信息，建议先固定一套策略（见第 4 节）。

## 2.4 superAGB_Doherty14

文件：

- superAGB_Doherty14/doherty14a_table1.txt

按块组织：

- 例如 7.0M Z=0.02 VW93
- 后续是 isotope 行

列示例：

- Species Yield MassExp ProdFact ...

与 AGB 一致建议：

- 若追求与 slug2 一致，用 MassExp（第 3 列）作为注入量。

## 2.5 isotope_data.txt

这个文件是同位素/反应网络基础数据，不是直接的 stellar yield 表。

在 Quokka 集成里它可用于：

1. 校验 isotope 名字合法性
2. 建立 isotope 索引映射

不建议把它直接当 channel yield 源。

## 3. 在 Quokka 中的推荐接入流程

## 步骤 1：先确定你要跟踪的同位素

例如：

- C12, N14, O16, Fe56

## 步骤 2：统一同位素命名

把不同表中的命名统一成同一套 key，例如：

- c12 -> C12
- fe56 -> Fe56
- p -> H1
- d -> H2

并处理特殊别名（在 Karakas 文件中可能出现）：

- al-6 -> Al26
- al*6 -> 跳过（同位素激发态别名）
- kr-5 -> Kr85
- kr*5 -> 跳过

## 步骤 3：把原始表转成 Quokka 统一中间表

建议你先做预处理脚本，生成规范化表（HDF5 或 CSV）：

1. SNII 表：Y_SNII[iso, M, Z]，可拆分为 wind/sn 两项。
2. AGB 表：Y_AGB[iso, M, Z, pmz, overshoot]。
3. superAGB 表：Y_superAGB[iso, M, Z]。

这样运行期只做插值，不做文本解析。

## 步骤 4：确定 AGB 文件选择策略

建议与 slug2 保持一致的默认策略：

1. PMZ 默认随质量变化：
- M <= 3: pmz = 2e-3
- 3 < M <= 4: pmz = 1e-3
- 4 < M < 5: pmz = 1e-4
- M >= 5: pmz = 0
2. overshoot：有可用模型时优先开。
3. 输入 Z 若不在离散点上，选择最近 Z（或做 logZ 插值）。

## 步骤 5：运行时查询

对每个星粒子，在给定 channel 下查询：

- Y_iso = interp(M_init, Z_birth, ...)

再按你的反馈模型转换成每步注入：

1. 事件注入（SNII）
- 在爆发步一次性注入 Y_iso。
2. 连续注入（WR/AGB）
- 这些表主要是积分产额，不天然给 dY/dt。
- 你需要额外给时间核 K(t)，例如：
  - top-hat
  - 指数衰减
  - 按寿命窗口归一化

满足：

- 积分后总量等于表给定总产额。

## 4. 给你一个最稳健的最小实现（MVP）

第一版建议：

1. 用 SNII_Sukhbold16 + AGB_Karakas16 + superAGB_Doherty14。
2. 都使用“抛射总量”语义：
- Sukhbold: wind/sn 列
- Karakas: mass(i)_lost
- Doherty: MassExp
3. 只做按质量和金属度插值，不做复杂时间核。
4. 先实现事件式注入（SNII + AGB终末一次性注入），验证守恒后再扩展连续注入。

## 5. 守恒检查（必须做）

每个测试末尾验证：

1. 全域注入同位素质量 = 星粒子抛射总质量（并行归约后比较）。
2. 通道和总量一致：
- chem_gas_iso = chem_ch_SNII_iso + chem_ch_AGB_iso + chem_ch_WR_iso（若启用 WR）。
3. 不出现负丰度（必要时 clamp 到 floor）。

## 6. 建议你接下来立即做的事

1. 先写一个离线预处理脚本，把四套原始文件规范化为统一 HDF5。
2. 在 Quokka 加一个 YieldTableManager（只负责加载和插值）。
3. 在 particle update/deposition 中只调用统一接口，不直接读文本。

## 7. 与你当前设计的对应关系

你前面定义的自动通道命名（例如 AGB-C12、SNII-C12）可以直接映射到这套表：

1. SNII-C12 <- SNII_Sukhbold16 或 Kobayashi0611 的 C12 插值值
2. AGB-C12 <- Karakas16/superAGB 的 C12 插值值
3. WR-C12 <- 需另备 WR 表或先用参数化核

这样就能无缝接到你在 Chuhan_test 里已经规划的同位素自动生成与逐星反馈框架。