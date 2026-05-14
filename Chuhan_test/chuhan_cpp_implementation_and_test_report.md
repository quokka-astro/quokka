#Chuhan_start 作用：记录本次 C++ 化学反馈实现范围、接线细节和测试结果。
# Chuhan C++ 实现流程与测试报告

## 1. 本次实现范围

本次在 Quokka C++ 主流程实现了以下能力：

1. 恒星粒子仅记录三通道步进化学反馈缓存（不再单独存 `birth_metallicity`）。
2. 新增逐星化学反馈三通道步进缓存：`SNII/WR/AGB`。
3. 新增网格沉积接口：支持总同位素字段 + 可选分通道字段沉积。
4. 在 `particleMeshInteraction` 中接入化学沉积阶段。
5. 为避免双计数：启用新化学反馈时，关闭 legacy SN `scalar_yield_per_SN` 注入分支。

## 2. 关键代码改动

### 2.1 粒子字段与参数扩展

文件：`src/particles/particle_types.hpp`

1. 在 `StochasticStellarPopParticleRealIdx` 新增：
- `metal_yield_snii_step`
- `metal_yield_wr_step`
- `metal_yield_agb_step`

2. 新增对应索引常量：
- `StochasticStellarPopParticleMetalYieldSNIIStepIdx`
- `StochasticStellarPopParticleMetalYieldWRStepIdx`
- `StochasticStellarPopParticleMetalYieldAGBStepIdx`

3. 更新 `StochasticStellarPopParticleRealComps`：
- 从 `14 (+ nGroups)` 扩展到 `17 (+ nGroups)`。

4. units 元数据补齐了上述字段单位。

5. 新增 `particles.*` 运行参数：
- `enable_chemical_feedback`
- `enable_SNII_metal`
- `enable_WR_metal`
- `enable_AGB_metal`
- `store_channel_fields`
- `chemical_scalar_offset`
- `chemical_num_scalars`
- `snii_metal_yield_fraction`
- `wr_metal_yield_rate_per_mass`
- `agb_metal_yield_rate_per_mass`
- `wr_age_start`, `wr_age_end`
- `agb_age_start`, `agb_age_end`
- `stellar_metallicity_fraction`
- `use_table_driven_chemical_yield`
- `chemical_yield_table_file`

### 2.2 星形成时初始化通道缓存

文件：`src/particles/particle_creation.hpp`

在 `ParticleCreationTraits<StochasticStellarPop>::ParticleCreator::operator()` 中：

1. 不再写入粒子级 `birth_metallicity`。
2. 初始化三通道步进缓存为 0。

### 2.3 每步更新中计算通道产额

文件：`src/particles/particle_update.hpp`

在 `ParticlePropertyUpdateTraits<ParticleType::StochasticStellarPop>::updateProperties(...)` 中：

1. 保持原有 luminosity 更新。
2. 每步先清零三通道缓存。
3. 若启用化学反馈：
- SNII：当 `SNProgenitor` 在本步跨过 `death_time`，注入
  `M_Z,SNII = (stellar_metallicity_fraction + yield_table_SNII_fraction) * mass_at_birth`。
- WR：在高质量星阶段且 `age` 落在 WR 时间窗内，注入
  `dM_Z,WR = wr_rate_from_table_or_param * mass_at_birth * dt`。
- AGB：在 `LowMassComposite` 阶段且 `age` 落在 AGB 时间窗内，注入
  `dM_Z,AGB = agb_rate_from_table_or_param * mass_at_birth * dt`。

### 2.4 网格化学沉积与通道字段

文件：`src/particles/particle_deposition.hpp`

1. 新增 `ChemicalFeedbackDeposition<...>(...)`：
- 对每个粒子读取三通道本步产额。
- 沉积到总化学字段（从 `chemical_scalar_offset` 开始，长度 `chemical_num_scalars`）。
- `store_channel_fields=true` 时，将 SNII/WR/AGB 分别沉积到后续 block。
- 沉积后清零粒子本步缓存，避免重复注入。

2. 修改 legacy SN 标量沉积：
- 在 `enable_chemical_feedback=true` 时跳过 `scalar_yield_per_SN` 旧路径，防止 SNII 金属双计数。

### 2.5 注册器与主流程接线

文件：`src/particles/PhysicsParticles.hpp`

1. 在 `PhysicsParticleDescriptorBase` 增加虚函数：
- `depositChemicalFeedback(...)`。

2. 在具体 descriptor 中实现：
- 仅 `StochasticStellarPop` 调用 `ChemicalFeedbackDeposition<...>`。

3. 在 `PhysicsParticleRegister` 增加：
- `depositChemicalFeedback(...)` 聚合调用。

文件：`src/simulation.hpp`

在 `particleMeshInteraction(...)` 中接入：

1. `createParticlesFromState(...)` 后
2. 调用 `particleRegister_.depositChemicalFeedback(...)`
3. 然后再执行 `depositSN(...)`

## 3. 字段布局约定（passive scalars）

设：
- `N = chemical_num_scalars`
- `O = chemical_scalar_offset`

则：

1. 总化学字段：`scalar[O ... O+N-1]`
2. 若 `store_channel_fields=true`：
- SNII block: 紧跟总字段后
- WR block: 紧跟 SNII block 后（若启用 WR）
- AGB block: 紧跟 WR block 后（若启用 AGB）

注意：当前实现会自动裁剪到 `numPassiveScalars` 上限，超出部分不会写入。

## 4. 测试方案与执行结果

### 4.1 编译测试

命令：

```bash
cmake --build build --target all -j4
```

结果：通过（无编译错误）。

### 4.2 CTest 回归（当前构建可用匹配）

命令：

```bash
ctest --test-dir build --output-on-failure -R "Particle|HydroShocktube"
```

结果：

1. `HydroShocktube` 通过
2. `HydroShocktubeCMA` 通过

说明：当前 build 配置中匹配到的是这两个测试；未匹配到粒子专用测试目标。

### 4.3 Python 产量管线回归

命令：

```bash
python3 scripts/python/test_chuhan_yield_pipeline.py
```

结果：`Ran 14 tests ... OK`。

## 5. 参数示例（可放入输入文件）

```toml
[particles]
enable_chemical_feedback = 1
enable_SNII_metal = 1
enable_WR_metal = 1
enable_AGB_metal = 1
store_channel_fields = 1
chemical_scalar_offset = 0
chemical_num_scalars = 1
stellar_metallicity_fraction = 0.014
use_table_driven_chemical_yield = 1
chemical_yield_table_file = "yields/chuhan_channel_yields.dat"
snii_metal_yield_fraction = 0.1
wr_metal_yield_rate_per_mass = 0.0
agb_metal_yield_rate_per_mass = 0.0
wr_age_start = 0.0
wr_age_end = 0.0
agb_age_start = 0.0
agb_age_end = 0.0
```

## 6. 现阶段实现性质

这是一个可运行的 C++ 主流程版本（MVP+主链接线），已满足：

1. 三通道步进注入字段与查表驱动产额
2. SNII/WR/AGB 多通道步进注入链路
3. 主流程调用与沉积防双计数

后续如要完全按表驱动（`yield[channel][isotope](mass, age, Z_birth, dt)`）可在现有框架上直接替换每步产额模型。
#Chuhan_end 作用：该报告用于追踪实现状态与后续扩展方向。
