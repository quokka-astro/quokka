# Chuhan 变更记录与测试方案

## 1. 本次记录目标

先把当前工作区中已经改动的文件列清楚，然后再按文件逐一补充：

1. 每个文件做了什么改动。
2. 改动之间的依赖关系和接入顺序。
3. 对应的测试方案、已执行测试和后续建议测试。

## 2. 当前已识别的改动文件

### 2.1 已修改的现有文件

1. `src/particles/PhysicsParticles.hpp`
2. `src/particles/particle_creation.hpp`
3. `src/particles/particle_deposition.hpp`
4. `src/particles/particle_types.hpp`
5. `src/simulation.hpp`

### 2.2 新增的文件

1. `scripts/python/chuhan_yield_pipeline.py`
2. `scripts/python/test_chuhan_yield_pipeline.py`
3. `src/particles/particle_chemical_yield.hpp`

## 3. 各文件具体做了什么

### 3.1 `src/particles/particle_types.hpp`

1. 引入了 `particles/particle_chemical_yield.hpp`，让粒子类型代码可以直接使用化学产额查表工具。
2. 新增了运行时 chemistry 布局配置：
	- `chemical_tracked_isotopes`
	- `chemical_tracked_channels`
3. 现在粒子类型层会把用户输入的通道/同位素列表自动拆分成有序向量，并在参数解析后刷新。
4. `StochasticStellarPop` 粒子的 real component 布局已经扩展为“固定粒子字段 + luminosity + 4 个 chemistry block”，为气体/恒星同构存储预留了空间。
5. `getParticleRealCompNames()` 和 `get_units_data()` 已改为基于运行时 chemistry 列表生成字段名和单位元数据，便于 Fields.yaml 与实际 chemistry 布局保持一致。
6. 新增了一组化学反馈运行参数，并在 `particleParmParse()` 中读取：
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
	- `wr_age_start`
	- `wr_age_end`
	- `agb_age_start`
	- `agb_age_end`
	- `use_table_driven_chemical_yield`
	- `chemical_yield_table_file`
	- `stellar_metallicity_fraction`
7. 如果启用表驱动化学产额，会尝试加载 `chemical_yield_table_file`，失败时打印警告并回退到参数化模式。

### 3.2 `src/particles/particle_creation.hpp`

1. 在 `StochasticStellarPop` 粒子创建时，把气体中的 chemistry block 直接复制到粒子 birth chemistry 区域，包含总 chemistry block 与各通道 block。
2. 这样星粒子和气体在 chemistry 布局上已经是同构的，后续沉积时可以直接按对应 block 读取出生丰度信息。

### 3.3 `src/particles/particle_deposition.hpp`

1. 在两个 legacy SN 沉积路径里加了条件判断：当 `enable_chemical_feedback` 开启时，不再执行原来的 `scalar_yield_per_SN` 标量沉积，避免和新化学反馈双计数。
2. 新增了 `ChemicalFeedbackDeposition(...)`：
	- 仅在化学反馈启用且容器存在时运行。
	- 直接根据粒子的质量、年龄、演化阶段和出生 chemistry block 现算 SNII / WR / AGB 的本步产额。
	- 把 chemistry 结果写回到被动标量区间中由 `chemical_scalar_offset` 和 `chemical_num_scalars` 指定的总 chemistry block 与各通道 block。
	- 现在已经改成按 tracked isotope 循环，并为每个通道分别调用 `queryYieldFraction(channel, isotope, mass, metallicity)`，因此总量、SNII、WR、AGB 都可以按元素独立查表。
	- baseline 项现在统一使用“恒星诞生时该 isotope 在总 birth chemistry block 里的丰度”，不再使用通道 block 值作为 baseline。
	- 如果没有启用表驱动产额，则会回退到原来的参数化标量率，但仍按每个 isotope 逐项注入。
3. 该函数以粒子所在单元为目标网格，使用体积倒数做密度化，并对被动标量组件边界做了保护。

### 3.4 `src/particles/PhysicsParticles.hpp`

1. 在 `PhysicsParticleDescriptorBase` 里新增了虚函数 `depositChemicalFeedback(...)`，作为新化学反馈路径的统一入口。
2. 在 `StochasticStellarPop` 对应的具体 descriptor 里实现了这个接口，并调用 `ChemicalFeedbackDeposition<...>(...)`。
3. 在 `PhysicsParticleRegister` 里增加了聚合入口 `depositChemicalFeedback(...)`，和已有的 `depositSN(...)`、`depositMass(...)` 等接口保持一致。
4. 这样主流程就可以对所有粒子类型统一调用，而真正有化学反馈的类型再自己处理。

### 3.5 `src/simulation.hpp`

1. 在 `particleMeshInteraction(...)` 中，把化学反馈沉积插入到 `createParticlesFromState(...)` 之后、`depositSN(...)` 之前。
2. 这样顺序上是先完成粒子生成，再把本步的连续/事件化学反馈写回网格，最后再走 SN 能量和动量沉积。
3. 代码里还加了注释，明确这一步负责沉积 WR / AGB / SNII 化学场。

### 3.6 `src/particles/particle_chemical_yield.hpp`

1. 新建了一个 C++ 化学产额查表模块，用来把 yield table 转成运行时可用的通道比例。
2. 这个头文件维护了一个上限为 4096 条记录的全局加载表，并用 `AMREX_GPU_MANAGED` 数组保存：
	- 质量
	- 金属丰度
	- SNII 比例
	- WR 比例
	- AGB 比例
3. 现在额外加入了 compact 的 `channel × isotope` 查询入口，可以直接按 `queryYieldFraction(channel, isotope, mass, metallicity)` 读取单个通道、单个同位素的产额。
4. 旧的总通道比例解析路径仍然保留，作为不带逐同位素表时的兼容回退。
5. 提供了基础工具函数：
	- 大小写归一化
	- 从同位素名提取元素符号
	- 判断元素是否属于金属元素
	- 从目录名或文件名中解析质量和金属丰度
6. 提供了几类表解析函数：
	- 旧式 5 列聚合表
	- `SNII_Kobayashi0611` 原始目录
	- `superAGB_Doherty14` 分块表
7. 提供了对外查询函数 `queryFractions(...)` / `queryYieldFraction(...)` / `loadTable(...)` 所依赖的查表数据准备逻辑，供粒子更新和沉积阶段按最近质量点、金属度和 isotope 选择产额。
8. 这个模块的目标不是直接做粒子沉积，而是把离线表整理成可被粒子更新与化学反馈沉积共同使用的运行时数据。

### 3.7 `scripts/python/chuhan_yield_pipeline.py`

1. 新建了一个 Python 管线脚本，用来把 yield 数据源整理成可验证、可查询、可导出的 manifest。
2. 先做输入标准化：
	- 同位素支持大小写和别名归一化，比如 `p -> H1`、`d -> H2`。
	- 支持逗号和空格混合的用户输入，并自动去重。
	- 通道名自动转成大写并去重。
3. 生成自动字段名：
	- `chem_gas_<iso>`
	- `chem_star_birth_<iso>`
	- `chem_ch_<CHANNEL>_<iso>`
4. 加入了 4 类数据解析函数，分别对应：
	- `SNII_Sukhbold16`
	- `SNII_Kobayashi0611`
	- `AGB_Karakas16`
	- `superAGB_Doherty14`
5. 增加了按“最近质量点 + 最近金属丰度”查询单个同位素产额的接口 `query_isotope_yield(...)`。
6. 增加了 `lookup_tracked_isotope_availability(...)`，可快速检查某个同位素在不同来源是否存在。
7. 增加了 `build_manifest(...)`，把 tracked isotopes、channels、自动字段和数据可用性汇总成 JSON 结构。
8. 最后提供 CLI 入口，可以直接生成 manifest 文件。

### 3.8 `scripts/python/test_chuhan_yield_pipeline.py`

1. 新建了对应的单元测试文件，覆盖上述 Python 管线的核心行为。
2. 测试内容包括：
	- 同位素标准化
	- 混合输入解析
	- 自动字段生成和去重
	- 4 类原始表解析
	- 同位素来源可用性检查
	- manifest 结构检查
	- CLI 端到端生成
	- `query_isotope_yield(...)` 的 4 类来源查询
3. 测试总数为 14 项，目标是验证脚本不仅能解析，还能稳定产出可用的 manifest 和查询结果。

## 4. 写作顺序

1. 先把粒子类型、更新、沉积和主流程这条 C++ 主线写清楚。
2. 再补 Python 管线和对应测试。
3. 最后整理测试方案，把已执行命令、覆盖范围和后续建议分开写。

## 5. 说明

1. `slug2/` 目录不纳入本次说明。
2. 这份笔记会继续作为完整变更记录使用，后面可以直接在这里追加测试方案和结论。

## 6. 如何测试

### 6.1 Python 管线测试（已执行）

1. 运行单元测试：

```bash
python3 scripts/python/test_chuhan_yield_pipeline.py
```

2. 本次实际结果：

```text
Ran 14 tests in 0.141s
OK
```

3. 该测试覆盖内容：
	- 同位素名标准化与别名。
	- 用户输入解析（逗号 + 空格混合、去重、大小写归一）。
	- 自动字段生成。
	- 四类产额表解析。
	- 可用性检查与 manifest 生成。
	- CLI 端到端生成。
	- `query_isotope_yield(...)` 四来源查询。

### 6.2 C++ 回归测试（已执行）

1. 运行与粒子和激波管相关的回归：

```bash
ctest --test-dir build --output-on-failure -R "Particle|HydroShocktube"
```

2. 本次执行状态：退出码为 0，说明该筛选范围内测试通过。

### 6.3 建议补充测试（建议后续执行）

1. 增加一个开启 `enable_chemical_feedback` 的最小输入用例，检查：
	- 总化学字段是否增长。
	- 分通道字段在 `store_channel_fields=true` 时是否增长。
2. 增加参数回退测试：
	- `use_table_driven_chemical_yield=true` 但路径无效时，确认打印警告并继续运行。
3. 增加双计数保护测试：
	- 开启化学反馈时，确认 legacy `scalar_yield_per_SN` 不再注入。
4. 增加步进一致性测试：
	- 连续两个 step 后，确认化学沉积由粒子的出生丰度和当前年龄重新计算，不依赖粒子临时缓存。

## 7. 如何使用

### 7.1 生成 manifest（已验证）

1. 运行命令：

```bash
python3 scripts/python/chuhan_yield_pipeline.py \
  --yields-dir yields \
  --isotopes C12,N14 Fe56 \
  --channels AGB,SNII WR \
  --output build/chuhan_manifest.json
```

2. 本次实际结果：

```text
Wrote manifest to build/chuhan_manifest.json
```

3. 输出文件用途：
	- 明确 tracked isotopes 与 channels。
	- 给出自动字段映射（gas / star_birth / channel）。
	- 给出各来源可用性，便于判断哪些同位素需要降级处理。

### 7.2 在 C++ 侧启用化学反馈

在输入参数里至少配置以下开关和范围：

1. 总开关与通道开关：
	- `particles.enable_chemical_feedback = 1`
	- `particles.enable_SNII_metal = 1`
	- `particles.enable_WR_metal = 1`
	- `particles.enable_AGB_metal = 1`

2. 被动标量布局：
	- `particles.chemical_scalar_offset = <总化学字段起始分量>`
	- `particles.chemical_num_scalars = <同位素数>`
	- `particles.store_channel_fields = 1`（如果需要分通道输出）

3. 表驱动配置：
	- `particles.use_table_driven_chemical_yield = 1`
	- `particles.chemical_yield_table_file = "yields"`

4. 时间窗与回退参数（建议保留）：
	- `particles.wr_age_start` / `particles.wr_age_end`
	- `particles.agb_age_start` / `particles.agb_age_end`
	- `particles.snii_metal_yield_fraction`
	- `particles.wr_metal_yield_rate_per_mass`
	- `particles.agb_metal_yield_rate_per_mass`

### 7.3 运行时执行顺序

当前主流程顺序是：

1. 先 `createParticlesFromState(...)`。
2. 再 `depositChemicalFeedback(...)`，把 SNII/WR/AGB 的本步化学产额沉积到网格。
3. 最后 `depositSN(...)` 做 SN 动量和能量反馈。

这意味着化学反馈与 SN 动力学反馈已经在同一时间步里完成接线，但保持了各自的沉积职责分离。
