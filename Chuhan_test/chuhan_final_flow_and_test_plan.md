# Chuhan 最终执行流程与测试方案

## 1. 已完成的可运行能力

当前已在 Python 管线中完成并通过测试：

1. 用户任意同位素输入解析
- 支持逗号/空格混合输入
- 支持大小写混合
- 自动标准化并去重

2. 用户任意通道输入解析
- 自动转大写并去重

3. 自动字段生成
- gas: chem_gas_<iso>
- star birth: chem_star_birth_<iso>
- channel: chem_ch_<CHANNEL>_<iso>

4. 自动查表可用性
- 对每个同位素自动检查四类来源是否存在
  - SNII_Sukhbold16
  - SNII_Kobayashi0611
  - AGB_Karakas16
  - superAGB_Doherty14

5. 自动查表查询接口（核心）
- query_isotope_yield(...)
- 按 source + isotope + mass + metallicity 自动选择最近表并返回产额

6. CLI 端到端输出
- 生成 manifest JSON
- 内含字段映射与同位素来源可用性

## 2. 代码位置

1. 主实现
- scripts/python/chuhan_yield_pipeline.py

2. 自动化测试
- scripts/python/test_chuhan_yield_pipeline.py

3. 相关说明笔记
- Chuhan_test/chuhan_yield_pipeline_usage_test_guide.md
- Chuhan_test/yields_usage_guide.md

## 3. 关键接口说明

## 3.1 输入标准化

1. normalize_isotope_name(raw)
- 例：c12 -> C12, p -> H1, d -> H2

2. parse_user_list(values, normalize_isotopes=False, uppercase=False)
- 支持 ["C12,N14", " Fe56 "] 这种混合输入
- 返回去重后的有序列表

## 3.2 自动字段

1. make_channel_fields(isotopes, channels)
- 输出 gas_fields / star_birth_fields / channel_fields

## 3.3 查表可用性

1. lookup_tracked_isotope_availability(yields_dir, tracked_isotopes)
- 输出每个 isotope 在四类来源中的 True/False 状态

## 3.4 自动查表（新增核心）

1. query_isotope_yield(yields_dir, source, isotope, mass_msun, metallicity, ...)

支持 source：

- SNII_Sukhbold16
- SNII_Kobayashi0611
- AGB_Karakas16
- superAGB_Doherty14

行为：

- 自动标准化 isotope
- 自动选择最近质量点
- 对需要 Z 的来源自动选择最近金属度
- 返回 source、selected_mass、selected_metallicity（若有）与 yield
- 对 Sukhbold 额外返回 sn/wind/total

## 3.5 manifest 汇总

1. build_manifest(yields_dir, tracked_isotopes, channels)

包含：

- tracked_isotopes
- channels
- auto_fields
- availability
  - sukhbold_masses
  - kobayashi_mass_grid
  - karakas_file_counts
  - doherty_blocks
  - tracked_isotope_sources

## 4. CLI 使用流程

## 4.1 生成 manifest

```bash
python3 scripts/python/chuhan_yield_pipeline.py \
  --yields-dir yields \
  --isotopes C12,N14 Fe56 \
  --channels AGB,SNII WR \
  --output build/chuhan_manifest.json
```

## 4.2 结果检查

1. 查看字段是否按输入自动生成
2. 查看 tracked_isotope_sources 是否满足预期
3. 对缺失来源的同位素做策略决定（禁用该通道或给 warning）

## 5. 测试方案（已执行）

当前测试覆盖 14 项，已全部通过。

## 5.1 已覆盖内容

1. 同位素标准化与别名
2. 动态字段生成与去重
3. 混合输入解析（逗号+空格）
4. 四类原始表解析
5. 自动查表可用性
6. manifest 结构与关键字段
7. CLI 端到端生成
8. query_isotope_yield 四来源查询

## 5.2 执行命令

```bash
python3 scripts/python/test_chuhan_yield_pipeline.py
```

本次结果：

- Ran 14 tests
- OK

## 6. 接入 Quokka C++ 主流程的建议步骤

1. 在初始化阶段读取 manifest
- 获得 tracked_isotopes/channels/auto_fields

2. 字段注册阶段
- 根据 auto_fields 映射到 passive scalar 索引

3. 粒子创建阶段
- 由你已有的 birth inheritance 方案写入 chem_star_birth_*

4. 反馈沉积阶段
- 对每个粒子按通道调用 query 逻辑对应的插值模块（C++版）
- 更新 chem_ch_* 与 chem_gas_*

5. 输出与回归
- plotfile/metadata 输出字段与映射
- 增加守恒测试与来源分解测试

## 7. 后续可优化项

1. 为 query_isotope_yield 增加可选插值模式
- nearest / linear / log-linear

2. 输出“缺失同位素来源”诊断报告
- 在 manifest 增加 missing_isotopes_by_source

3. 统一离线预处理
- 将文本表一次转换为 HDF5，运行期只做快速插值

## 8. 结论

你要求的核心目标“用户写入同位素后自动查表、自动生成字段、并可测试验证”已经在当前 Python 管线中完成并通过测试。

下一步只需把这套 manifest + query 语义迁移到 Quokka C++ 运行时即可。