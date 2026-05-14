# 同位素自动通道化学跟踪设计笔记（Gas + Star）

## 1. 目标

实现一个“声明式同位素跟踪”系统：

- 用户仅在输入参数里声明要跟踪的同位素，例如：`C12, N14, Fe56`。
- 框架自动创建并维护：
  - 气体化学场（isotope abundance in gas）
  - 恒星粒子化学字段（birth abundance + ejecta bookkeeping）
  - 反馈通道场（例如 `AGB-C12`, `SNII-C12`, `WR-C12` 等）
- 运行时自动把各通道产额沉积到对应同位素通道，支持后处理时按通道分解。

## 2. 核心需求拆解

1. Gas 与 Star 都要有化学信息。
2. 只需指定 isotope list，系统自动派生通道字段。
3. 对每个 isotope 自动创建：
   - `AGB-<iso>`
   - `SNII-<iso>`
   - `WR-<iso>`
4. 支持同位素丰度持续跟踪（守恒量语义）。

## 3. 推荐数据模型

## 3.1 规范化命名

建议内部统一使用下划线命名，避免 `-` 在某些输出工具中的兼容问题：

- `chem_gas_C12`
- `chem_gas_N14`
- `chem_gas_Fe56`

- `chem_ch_AGB_C12`
- `chem_ch_SNII_C12`
- `chem_ch_WR_C12`

- `chem_star_birth_C12`
- `chem_star_birth_N14`
- `chem_star_birth_Fe56`

可在输出层保留用户友好的别名（例如 metadata 中显示 `AGB-C12`）。

## 3.2 字段分层

建议分三层：

1. `Gas inventory`：当前气体同位素守恒密度（主状态量）。
2. `Channel tracers`：按反馈源分解的守恒密度（AGB/SNII/WR）。
3. `Star properties`：恒星粒子的出生丰度、累计抛射量、可选剩余可抛质量。

## 4. 参数接口设计

建议新增参数组：`chemistry_feedback` 与 `particles`。

示例：

```toml
[chemistry_feedback]
enable = true
tracked_isotopes = ["C12", "N14", "Fe56"]
channels = ["AGB", "SNII", "WR"]
store_channel_fields = true
store_star_birth_abundance = true

[particles]
enable_chemical_feedback = 1
```

行为：

- 读取 `tracked_isotopes` 后自动计算需要的 scalar 数量。
- 根据 `channels` 自动生成通道字段。
- 若 `store_channel_fields=false`，则仅存总同位素，不存分通道（省内存）。

## 5. 架构改造建议

## 5.1 新增“化学字段注册器”

新增模块（建议）：

- `src/chemistry/ChemicalFieldRegistry.hpp`
- `src/chemistry/ChemicalFieldRegistry.cpp`（或 header-only）

职责：

1. 解析 isotope 与 channels。
2. 生成字段描述（名称、units、索引）。
3. 暴露索引查询接口：
   - `gasIsotopeIndex("C12")`
   - `channelIndex("AGB", "C12")`
   - `starBirthIndex("C12")`

## 5.2 与 Hydro passive scalars 的对接

关键思路：

- `Physics_Traits<problem_t>::numPassiveScalars` 目前多为编译期常量。
- 若要“完全运行时动态创建字段”，需重构为运行时变量，影响较大。

建议两阶段方案：

1. 第一阶段（低风险）
- 预留 `max_chem_scalars` 编译期上限。
- 运行时根据 isotope list 激活前 N 个，并写入 varnames。

2. 第二阶段（高自由度）
- 推动 hydro scalar 组件支持运行时可变组件数。

## 5.3 粒子字段扩展

修改：

- `src/particles/particle_types.hpp`
- `src/particles/particle_creation.hpp`
- `src/particles/particle_IO.hpp`

建议：

1. 在星粒子 real data 增加“动态化学块”概念（建议固定上限 + 运行时有效长度）。
2. 至少包括：
   - birth abundance per isotope
   - cumulative ejecta per channel-isotope（可选）
3. 新生恒星时，从母气体单元读取同位素质量分数，写入 `star_birth_*`。

## 5.4 反馈沉积接口升级

修改：

- `src/particles/particle_deposition.hpp`
- `src/particles/particle_update.hpp`
- `src/particles/PhysicsParticles.hpp`
- `src/simulation.hpp`

建议：

1. 将现有单一 `scalar_yield_per_SN` 扩展为：
   - `yield[channel][isotope](mass, age, Z_birth, dt)`
2. 每步计算并沉积到：
   - `chem_gas_<iso>`（总）
   - `chem_ch_<channel>_<iso>`（分通道）
3. 同一星群同一步可并行贡献多通道。

## 6. 自动生成规则（用户视角）

输入：`tracked_isotopes = [C12, N14, Fe56]`, `channels=[AGB,SNII,WR]`

系统自动生成：

1. Gas 总同位素：
- `chem_gas_C12`
- `chem_gas_N14`
- `chem_gas_Fe56`

2. 通道同位素：
- `chem_ch_AGB_C12`, `chem_ch_AGB_N14`, `chem_ch_AGB_Fe56`
- `chem_ch_SNII_C12`, `chem_ch_SNII_N14`, `chem_ch_SNII_Fe56`
- `chem_ch_WR_C12`, `chem_ch_WR_N14`, `chem_ch_WR_Fe56`

3. Star 出生字段：
- `chem_star_birth_C12`
- `chem_star_birth_N14`
- `chem_star_birth_Fe56`

## 7. 产量表与查表

建议按 channel 组织独立表：

- `yield_AGB.h5/csv`
- `yield_SNII.h5/csv`
- `yield_WR.h5/csv`

每张表支持多 isotope 输出列：

- 输入维度：`M_init, age, (optional Z_birth)`
- 输出：`dM_C12, dM_N14, dM_Fe56, ...`

实现建议：

- GPU 侧使用紧凑张量布局：`yield[channel][iso][m_idx][t_idx]`
- 统一插值器接口，避免每通道重复代码。

## 8. 守恒与语义

必须明确所有字段是“守恒密度”还是“质量分数”：

1. Hydro 中建议使用“同位素守恒密度”（与现有 scalar 语义一致）。
2. 粒子 birth 字段建议用“质量分数”。
3. 沉积时以质量守恒闭合：
- `sum(cell_delta_iso_mass) = sum(particle_ejecta_iso_mass)`

## 9. I/O 与可诊断性

需要保证以下输出可见：

1. plotfile varnames 自动包含生成的化学字段。
2. particle plotfile 自动包含 `chem_star_birth_*`。
3. metadata 写入：
- isotopes 列表
- channels 列表
- 字段映射表（name -> component index）

## 10. 测试设计

## 10.1 字段生成测试

新增测试：

- `tracked_isotopes=[C12,N14,Fe56]` 时，断言字段总数与命名完整。

## 10.2 继承测试（Gas -> Star）

- 设置已知单元同位素质量分数。
- 触发造星。
- 检查新粒子 `chem_star_birth_*` 与单元分数一致。

## 10.3 通道沉积测试

- 手动构造一颗粒子，给定已知 AGB/SNII/WR 产额。
- 运行一步后检查：
  - `chem_ch_*` 分量与预期一致。
  - `chem_gas_* = 各通道和`。

## 10.4 守恒测试

- 全域归约检查同位素质量守恒误差。
- MPI/GPU 下要求误差在浮点可接受范围。

## 11. 预计改动文件（最小集合）

- `src/chemistry/ChemicalFieldRegistry.hpp`（新增）
- `src/chemistry/ChemicalFieldRegistry.cpp`（新增，可选）
- `src/particles/particle_types.hpp`
- `src/particles/particle_creation.hpp`
- `src/particles/particle_update.hpp`
- `src/particles/particle_deposition.hpp`
- `src/particles/PhysicsParticles.hpp`
- `src/simulation.hpp`
- `src/QuokkaSimulation.hpp`（变量名/plotfile 字段注册）

## 12. 实施建议（分期）

第一期（MVP）：

1. 支持 isotope list + 自动生成 gas 总同位素字段。
2. 支持 star birth abundance 继承。
3. 支持 SNII 单通道多同位素注入。

第二期：

1. 增加 AGB/WR 通道。
2. 增加 `chem_ch_*` 分通道字段。

第三期：

1. 增加 Z_birth 依赖产量表。
2. 优化内存（可选压缩或按需输出）。

## 13. 风险与注意

1. 动态字段数量与编译期 scalar 设计存在结构性冲突，需要“预留上限 + 运行时激活”过渡方案。
2. 新增粒子字段会影响 checkpoint 兼容性，需版本标识。
3. 通道字段数量为 `N_isotope * N_channel`，内存开销需评估。
4. GPU 原子加写入大量字段时可能影响性能，建议按 tile 局部缓存后归并。

## 14. 总结

该任务可以落地为“同位素声明驱动的自动字段系统”：

- 用户只给 isotope list。
- 框架自动生成 gas/stellar/channel 字段。
- 反馈模块按 channel x isotope 沉积。
- 输出可直接做丰度与来源分解分析（AGB vs SNII vs WR）。