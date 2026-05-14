# Quokka 逐星化学反馈实施笔记（SNII + WR + AGB）

## 1. 需求重述

目标是在现有粒子-网格耦合框架中加入逐星化学反馈，满足：

1. SNII 发生时，同步注入金属（事件型注入）。
2. WR 和 AGB 演化阶段，在每个系统步中持续注入金属（连续型注入）。
3. 同一个星群粒子在同一步可以同时触发多个产出通道（例如 WR 连续损失 + SNII 事件）。
4. 各通道产量来自独立产量表，按恒星质量与年龄查表。

## 2. 现状代码链路（已确认）

主时序在 src/simulation.hpp：

- 时间推进后调用 particleRegister_.updateParticleProperties(cur_time, dt_[0])。
- 然后调用 particleMeshInteraction(cur_time, dt_[0])。
- particleMeshInteraction 内会调用 particleRegister_.depositSN(...)。

关键文件：

- src/simulation.hpp
- src/particles/PhysicsParticles.hpp
- src/particles/particle_deposition.hpp
- src/particles/particle_update.hpp
- src/particles/particle_types.hpp

当前现状：

1. SN 沉积已存在，入口为 SNDeposition(...)，并且已经支持把产量加到 passive scalar0（通过 particles.scalar_yield_per_SN，单通道单标量）。
2. WR/AGB 没有现成功能。
3. 粒子阶段枚举为 HighMassNonExploding, SNProgenitor, SNRemnant, LowMassComposite, Removed，不能表达 WR/AGB 活动窗口。
4. 粒子属性更新机制已有模板框架（ParticlePropertyUpdateTraits），当前主要用于光度更新，可复用为“每步化学反馈预计算”。

## 3. 建议的目标架构

采用“双路径 + 单汇总”设计：

1. 事件通道（SNII）
- 仍沿用现有 depositSN 路径。
- 将 SNII 产量从固定常数改为查表（质量 + 年龄）。

2. 连续通道（WR/AGB）
- 在每个系统步，根据星粒子质量和年龄计算 WR/AGB 的 dM_metal/dt。
- 乘 dt 得到本步产量，再沉积到网格。

3. 单汇总思想
- 对每个粒子分别计算三通道贡献，然后加和到本步沉积缓冲区。
- 这样天然支持“同一粒子同一步多通道同时产出”。

## 4. 数据模型与参数设计

### 4.1 输入参数（建议新增）

在 particles 参数组新增：

- particles.enable_chemical_feedback = 1
- particles.enable_SNII_metal = 1
- particles.enable_WR_metal = 1
- particles.enable_AGB_metal = 1

- particles.snii_yield_table = <path>
- particles.wr_yield_table = <path>
- particles.agb_yield_table = <path>

- particles.chemical_output_mode = scalar_density
- particles.chemical_scalar_offset = 0
- particles.chemical_num_scalars = 1

说明：

- 第一阶段先支持 1 个金属总量标量（映射到 scalar0 或指定 offset）。
- 后续可扩展到多元素（C/O/Fe）多标量。

### 4.2 产量表格式建议

先采用统一二维表：

- 轴 1: 初始质量 M_init（单位 Msun）
- 轴 2: 年龄 t_age（单位 yr）
- 值: 通道金属产量率或累计产量

建议支持两种表意：

1. rate 型: y(M, t) = dM_metal/dt
2. cumulative 型: Y(M, t) = 累计金属产量

实现时统一转换为步进注入量：

- rate: DeltaM = y(M, t_mid) * dt
- cumulative: DeltaM = Y(M, t+dt) - Y(M, t)

## 5. 代码修改清单（按优先级）

### A. 新增化学产量表模块

新增文件：

- src/particles/particle_chemical_yield.hpp

职责：

1. 读入 SNII/WR/AGB 三张表。
2. 提供 GPU 可访问常量表（参考 luminosity table 的组织方式）。
3. 提供统一查询接口：
   - querySNIIYield(mass, age, dt)
   - queryWRYield(mass, age, dt)
   - queryAGBYield(mass, age, dt)

可参考：

- src/particles/particle_update.hpp 中 LuminosityGpuConstTables 的加载和设备侧使用模式。

### B. 扩展粒子属性更新（每步计算通道产出）

修改文件：

- src/particles/particle_update.hpp
- src/particles/particle_types.hpp

建议做法：

1. 在 StochasticStellarPop 粒子 real 分量中新增“本步通道产出缓存”（至少 3 个）：
   - metal_yield_snii_step
   - metal_yield_wr_step
   - metal_yield_agb_step

2. 在 ParticlePropertyUpdateTraits<StochasticStellarPop>::updateProperties(...) 中：
   - 计算 age = current_time - birth_time
   - 依据 mass_at_birth 和 age 查三张表
   - 写入本步缓存（单位 g）

这样好处：

- 将“查表逻辑”与“沉积逻辑”解耦。
- 后续调试可直接输出粒子本步产量。

### C. 扩展沉积路径（网格注入）

修改文件：

- src/particles/PhysicsParticles.hpp
- src/particles/particle_deposition.hpp
- src/simulation.hpp

建议新增接口：

- 在 PhysicsParticleDescriptorBase 加入 virtual depositChemicalFeedback(...)
- 在 StochasticStellarPop 描述符实现该接口

在 particleMeshInteraction(...) 中顺序建议：

1. computeSinkAccretion
2. applySinkAccretion
3. createParticlesFromState
4. depositChemicalFeedback  (WR/AGB 连续 + SNII 化学)
5. depositSN                (SN 动力学/热能/动量，及阶段跃迁)
6. destroyParticles

说明：

- 若希望 SNII 金属与 SN 动力学严格同一事件同一位置，可将 SNII 金属注入并入现有 SNDeposition 的 SN 爆发分支。
- WR/AGB 连续项建议单独函数，例如 depositContinuousChemicalFeedback(...)。

### D. SNII 与阶段逻辑一致性

修改文件：

- src/particles/particle_deposition.hpp
- src/particles/particle_types.hpp

原则：

1. SNII 事件判据继续由 SNProgenitor -> SNRemnant 的跃迁控制（已有）。
2. WR/AGB 不必须依赖 evolution stage，可由质量-年龄窗口决定是否有非零产量。
3. 为避免重复注入，SNII 金属只在“跃迁发生步”注入一次。

## 6. 同一步多通道并存的算法约定

对任一粒子 p，在一步 [t, t+dt]：

- DeltaM_Z_SNII = f_SNII(M_init, age, dt, trigger_SNII)
- DeltaM_Z_WR   = f_WR(M_init, age, dt)
- DeltaM_Z_AGB  = f_AGB(M_init, age, dt)

总注入量：

- DeltaM_Z_total = DeltaM_Z_SNII + DeltaM_Z_WR + DeltaM_Z_AGB

沉积时：

- 使用与 SN 一致的空间核（或独立核）分配到周围网格。
- 对每个格点以原子加方式写入 scalar 分量，保持 GPU 安全。

## 7. 测试方案（分层）

### 7.1 单元测试：查表正确性

新增测试建议：

- src/problems/ParticleChemicalYield/testParticleChemicalYield.cpp

验证项：

1. 给定质量和年龄，SNII/WR/AGB 查询值与参考值一致。
2. 表边界行为（外推/截断）符合预期。
3. rate 型与 cumulative 型在小 dt 下结果一致。

### 7.2 组件测试：沉积守恒

新增测试建议：

- src/problems/ParticleChemicalFeedback/testParticleChemicalFeedback.cpp

构造：

1. 单颗粒子固定在域中心，无流动背景。
2. 手动设置质量与年龄，令三通道都非零。
3. 运行 1-3 步后检查：
   - 网格中新增总金属质量 = 粒子通道注入总和（并行归约后比较）
   - 若设计为从粒子质量扣除，则验证总质量守恒闭合。

### 7.3 回归测试：与现有 DiskGalaxy 场景联动

可复用：

- src/problems/DiskGalaxy/testDiskGalaxy.cpp

做法：

1. 加入新参数开关，默认关闭，不影响现有回归。
2. 新增一个开启 WR/AGB 的回归输入，检查：
   - 金属标量总体单调上升。
   - SNII-only 与 SNII+WR+AGB 的差异在预期范围。

### 7.4 CTest 集成

需要同步修改：

- 对应问题目录下 CMakeLists.txt（新增测试 target）
- regression/quokka-tests.ini（新增回归项）
- inputs/ 下新增对应 toml

## 8. 开发顺序（推荐迭代）

第 1 迭代（最小可用）：

1. 只做单标量 scalar0。
2. 只做 SNII 查表替换（先不加 WR/AGB）。
3. 跑通现有 SN 相关测试，确保不回退。

第 2 迭代：

1. 接入 WR/AGB 连续注入。
2. 增加同一步三通道累加逻辑。
3. 新增沉积守恒测试。

第 3 迭代：

1. 扩展多元素多标量。
2. 增加 metadata/plotfile 输出字段（每通道累计注入量）。

## 9. 风险与注意事项

1. AMR 子循环：当前 particleMeshInteraction 在注释中已提示需要完善 subcycling 兼容，化学反馈也要遵循同样时间积分语义。
2. GPU Lambda 捕获：遵循项目规范，避免在 device lambda 中捕获 host 指针，使用 GpuArray/设备常量表。
3. 双重注入风险：若 SNII 金属既在 continuous 逻辑中算一次、又在 SN 事件中算一次，会导致翻倍，必须明确只在事件触发步注入一次。
4. 标量守恒：若金属以 conserved scalar density 存储，更新应与总质量一致；必要时加入 floors/clip。

## 10. 验证命令建议

构建与测试：

1. cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3
2. cmake --build build --target all
3. ctest --test-dir build --output-on-failure
4. ctest --test-dir build -R ParticleChemical

静态检查：

1. scripts/tidy.sh build changed

## 11. 结论

现有框架已经具备三件关键基础：

1. 粒子每步属性更新入口（可做查表）。
2. 粒子-网格沉积入口（可做连续化学注入）。
3. SN 事件沉积框架（可挂接 SNII 事件金属注入）。

因此可以在不推翻现有架构的前提下完成 SNII + WR + AGB 三通道逐星化学反馈，并支持同一星群在同一步多通道并发产出。
