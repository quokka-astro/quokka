# Chuhan 修改说明（简化版）

## 已修改的文件

### src/particles/PhysicsParticles.hpp
这个文件增加了化学反馈沉积的统一接口（`depositChemicalFeedback(...)`）。
作用是把“主流程调用”和“具体粒子类型实现”解耦，让只有需要化学反馈的粒子去执行对应沉积逻辑。
同时在粒子注册层提供聚合入口，调用方式与已有 SN/质量沉积接口保持一致。

### src/particles/particle_creation.hpp
这个文件在创建 `StochasticStellarPop` 粒子时，把气体中的 chemistry block 拷贝到粒子的 birth chemistry 区域。
作用是保留恒星诞生时的化学组成，作为后续 WR/AGB/SNII 化学产额计算的基线。
这样星粒子与气体在化学字段布局上更一致，后续读写更直接。

### src/particles/particle_deposition.hpp
这个文件是化学反馈沉积的核心接入点，新增了 `ChemicalFeedbackDeposition(...)`。
它按粒子质量、年龄、通道与同位素逐项计算产额，并把结果沉积到被动标量对应区间。
另外还在 legacy SN 标量沉积路径上做了开关保护，避免开启化学反馈时出现双计数。

### src/particles/particle_types.hpp
这个文件扩展了粒子类型层的 chemistry 运行时配置与参数解析。
作用包括：维护 tracked isotopes/channels、扩展 `StochasticStellarPop` 的 real component 布局、动态生成字段名和单位元数据。
还加入了表驱动化学产额相关开关与文件加载逻辑，加载失败时可回退到参数化模式。

### src/simulation.hpp
这个文件在主时间步流程里接入了化学反馈沉积调用。
当前顺序是先创建粒子，再做化学反馈沉积，最后执行 SN 能量/动量反馈。
作用是把化学反馈与动力学反馈都接到同一步内，同时保持职责分离。

## 新增的文件

### src/particles/particle_chemical_yield.hpp
这是 C++ 侧新增的化学产额查表模块。
它负责加载/整理产额表，并提供按通道、同位素、质量、金属丰度查询比例的接口（如 `queryYieldFraction(...)`）。
作用是把离线产额数据转成运行期可直接调用的统一查表能力，供粒子更新与沉积共用。
