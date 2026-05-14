# 恒星诞生继承形成气体化学信息：定位与实施笔记

## 1. 需求

在恒星粒子形成时，让粒子继承其形成母气体单元的化学信息（至少金属度；可扩展到多元素组分）。

## 2. 代码定位结论

### 2.1 粒子创建主路径

- 调用入口：`createParticlesFromState(...)`
  - 文件：src/particles/PhysicsParticles.hpp
- 具体实现：`ParticleCreationTraits<ParticleType::StochasticStellarPop>::createParticles(...)`
  - 文件：src/particles/particle_creation.hpp
- 核心写入逻辑：`ParticleCreator::operator()(...)`
  - 文件：src/particles/particle_creation.hpp

### 2.2 当前行为（已确认）

在 `particle_creation.hpp` 的 StochasticStellarPop 创建逻辑中：

1. 会设置粒子质量、速度、birth/death 元数据、evolution stage、mass_at_birth。
2. 在从网格扣除形成质量时，会对网格 `density/momentum/energy/scalars` 乘同一 `factor`：
   - `state_arr(i,j,k, scalar0_index + nn) *= factor`
3. 但是不会把母气体的化学标量写入粒子字段。

结论：

- 目前“气体化学守恒扣减”存在。
- 目前“新生恒星继承化学成分”不存在。

### 2.3 字段层面的根因

`StochasticStellarPopParticleRealIdx` 目前只有：

- `mass, vx, vy, vz, birth_time, death_time, birth_x/y/z, death_x/y/z, death_density, mass_at_birth, luminosity...`

没有任何“出生化学信息”字段（如 birth_metallicity / birth_scalar[n]）。

对应定义文件：

- src/particles/particle_types.hpp

## 3. 设计建议

建议采用“两层存储”策略：

1. **最小版本（推荐先做）**
- 仅继承一个标量：`birth_metallicity`（或 `birth_scalar0_mass_fraction`）。
- 用于后续 SNII/WR/AGB 产量修正与统计分析。

2. **扩展版本**
- 继承多个化学分量：`birth_scalar0 ... birth_scalarN-1`。
- 支持多元素产量表（C/O/Fe 等）与初始成分依赖。

## 4. 具体改动文件清单

### A. 粒子字段扩展

修改：

- src/particles/particle_types.hpp

建议新增（StochasticStellarPop real 分量）：

- `birth_metallicity`（最小版本）
- 或 `birth_scalar0` 到 `birth_scalarK`（扩展版本）

并同步：

- 对应 `constexpr` index alias
- `StochasticStellarPopParticleRealComps` 计数
- 粒子单位表（如果有 units map 输出）

### B. 粒子创建时拷贝母气体化学信息

修改：

- src/particles/particle_creation.hpp

位置：

- `ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator::operator()`

建议逻辑（每个新粒子创建时）：

1. 从当前网格单元读取：
- `rho_gas = state_arr(i,j,k,density_index)`
- `rho_scalar_n = state_arr(i,j,k,scalar0_index+n)`
2. 计算质量分数：
- `X_n = rho_scalar_n / rho_gas`（需防 `rho_gas <= floor`）
3. 写入粒子出生字段：
- `p.rdata(birth_scalarN_idx) = X_n` 或 `birth_metallicity`

注意：

- 该写入应发生在更新网格状态（乘 factor）之前或之后都可，但必须以“形成时刻母气体成分”为准。
- 推荐在创建粒子并设置基础属性后、网格扣减前进行，语义最清晰。

### C. 输出与重启兼容

修改（按需）：

- src/particles/particle_IO.hpp

确认事项：

1. 新增字段自动进入 plotfile/checkpoint 的 real component names。
2. 单位映射补全（无量纲质量分数可设 dimensionless）。

## 5. 与化学反馈模块的接口约定

后续你要做逐星化学反馈时，可直接使用出生成分：

- `Z_birth = p.rdata(birth_metallicity_idx)`
- 或 `X_birth[element]`

用于：

1. 选择/修正 SNII、WR、AGB 产量表（若产量依赖初始金属度）。
2. 做风损失金属组成配比。
3. 输出“按出生金属度分箱”的反馈统计。

## 6. 测试方案

### 6.1 单元/组件测试（建议新增）

新增测试问题：

- src/problems/ParticleBirthChemInheritance/testParticleBirthChemInheritance.cpp

构造：

1. 单网格或少量网格，给定已知 `rho` 与 `scalar0..N`。
2. 触发一次星形成。
3. 读取新生粒子 real data，检查出生字段是否等于 `rho_scalar/rho`。
4. 同时验证网格 scalar 的扣减仍保持原有守恒语义。

### 6.2 回归测试

- 在现有有星形成的问题上增加开关：
  - `particles.inherit_birth_chemistry = 0/1`
- 验证开启前后：
  1. 动力学主量不应出现非预期漂移。
  2. 粒子输出里新增字段存在且取值合理。

## 7. 参数建议

在 `particles` 参数组增加：

- `particles.inherit_birth_chemistry = 1`
- `particles.inherit_birth_chemistry_mode = mass_fraction`（或 `partial_density`）
- `particles.inherit_birth_num_scalars = 1`（后续可扩展）

## 8. 实施顺序（最稳妥）

1. 先加 `birth_metallicity` 单字段。
2. 在创建时写入 `scalar0 / rho`。
3. 打通 I/O 与 checkpoint。
4. 加单元测试。
5. 再扩展到多 scalar 继承。

## 9. 关键风险

1. 粒子 real 分量数量变化会影响重启兼容；需明确旧 checkpoint 是否兼容。
2. 若 `numPassiveScalars == 0`，应自动禁用该功能并给出清晰日志。
3. `rho` 过小需防除零，建议使用密度 floor。

## 10. 结论

你要的功能在当前代码中尚未实现，但改造路径很直接：

- 在 `particle_types.hpp` 增加“出生化学字段”；
- 在 `particle_creation.hpp` 形成粒子时从母气体单元拷贝化学信息到粒子；
- 增加对应测试与输出字段。

这样可以无缝衔接你后续的 SNII/WR/AGB 逐星化学反馈。