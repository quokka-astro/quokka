# ChuhanProblem 新配置模拟说明

**配置时间**: 2026年4月25日 22:57  
**状态**: ✅ 已配置并编译中

---

## 🔧 新配置参数

根据用户要求,已修改以下参数来增大模拟时间以观察超新星事件:

###时间参数修改

| 参数 | 修改前 | 修改后 | 倍数 |
|------|--------|--------|------|
| `initial_dt` | 1.0e13 秒 | 2.0e13 秒 | ×2 |
| `max_timesteps` | 10 | 100 | ×10 |
| 总模拟时长 | 3.16 Myr | 63.2 Myr | ×20 |

### 配置文件

**文件位置**: `/Users/meow/quokka/inputs/ChuhanProblem.toml`

**关键参数**:
```toml
# 时间参数
max_timesteps = 100                      # 最大时间步数
initial_dt = 2.0e13                      # 初始时间步大小 (秒)
plotfile_interval = 10                   # 输出间隔
checkpoint_interval = 10                 # 检查点间隔

# 物理参数
problem.n0 = 1.0e5                       # 环境密度 (g/cm³)
problem.Tamb = 1.0e1                     # 环境温度 (K)
particles.SN_scheme = "SN_thermal_or_thermal_momentum"  # SN反馈
particles.enable_SNII_metal = 1          # 启用 SNII 金属反馈
particles.enable_WR_metal = 1            # 启用 WR 金属反馈
particles.enable_AGB_metal = 1           # 启用 AGB 金属反馈
```

---

## 📊 预期结果

### 时间覆盖范围
- **旧配置**: 0 ~ 1.59 Myr (基于11个时间步)
- **新配置**: 0 ~ 63.2 Myr (基于100个时间步,每步 0.632 Myr)

### 预期输出
- **Plotfile 数量**: 约 100 个 (vs 原来的 11 个)
- **密度切片图**: 约 100 张 (vs 原来的 11 张)
- **化学标量图**: 约 300 张 (3种 × 100张)
- **文件大小**: 显著增加 (需要充足磁盘空间)

### 超新星事件预期
根据恒星演化理论:

| 恒星类型 | 生命周期 | 预期 SN 时间 |
|---------|---------|-------------|
| O 星 (>30 M☉) | 数百万年 | 3-5 Myr |
| B 星 (10-30 M☉) | 数百万年 | 10-20 Myr |
| A 星 (3-10 M☉) | 数百万年 | 20-50 Myr |

**结论**: 在 63.2 Myr 的覆盖范围内,应该能够观察到大部分恒星的超新星事件!

---

## 🏗️ 编译状态

### 构建系统重新配置

1. **发现问题**: 原始构建配置为 1D,而 ChuhanProblem 需要 3D
   
2. **解决方案**: 重新配置为 3D
   ```bash
   cmake -S /Users/meow/quokka -B /Users/meow/quokka/build \
     -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3
   ```

3. **编译过程**:
   - ✅ CMake 配置完成
   - ⏳ Ninja 编译中 (使用多核编译加速)
   - 预计完成时间: 10-30 分钟

### 构建命令

```bash
/Users/meow/.local/bin/ninja -C /Users/meow/quokka/build \
  src/problems/ChuhanProblem/all -j 2
```

---

## 🚀 运行流程

### 步骤 1: 等待编译完成
编译完成后,可执行文件将位于:
```
/Users/meow/quokka/build/src/problems/ChuhanProblem
```

### 步骤 2: 运行模拟
```bash
cd /Users/meow/quokka/tests
/Users/meow/quokka/build/src/problems/ChuhanProblem \
  ../inputs/ChuhanProblem.toml
```

### 步骤 3: 分析结果
模拟完成后,执行分析:
```bash
/Users/meow/quokka/.venv/bin/python \
  /Users/meow/quokka/run_full_analysis.py
```

---

## 📁 重要文件

### 配置文件
- ✅ `/Users/meow/quokka/inputs/ChuhanProblem.toml` - 已修改

### 分析脚本
- ✅ `/Users/meow/quokka/run_full_analysis.py` - 完整分析工具
- ✅ `/Users/meow/quokka/compile_and_run.sh` - 自动化脚本

### 输出目录
- 📊 `/Users/meow/quokka/tests/chuhan_density_sn_plots/` - 密度图 (~100张)
- 📊 `/Users/meow/quokka/tests/chuhan_metallicity_plots/` - 化学图 (~300张)
- 📊 `/Users/meow/quokka/tests/sn_analysis_plots.png` - 综合分析

---

## 💡 关键改进

### vs. 旧配置的优势

| 方面 | 旧配置 | 新配置 | 改进 |
|------|--------|--------|------|
| 模拟时长 | 1.59 Myr | 63.2 Myr | 40倍 |
| 时间步数 | 11 | 100 | 9倍 |
| 每步时间 | 0.159 Myr | 0.632 Myr | 4倍 |
| **SN 观测概率** | 低 | **高** | **显著** |
| 数据量 | 小 | 大 | 大 |

### 期望的新现象

1. **超新星事件**: 应该能在 20-50 Myr 之间观察到首个 SN
2. **化学丰度增长**: Scalar 值应从 0 增加到非零值
3. **密度扰动**: 气体密度应在 SN 事件处出现变化
4. **粒子演化**: 恒星应显示年龄增长和质量损失迹象

---

## ⏱️ 时间线预估

| 任务 | 时间 | 状态 |
|------|------|------|
| 编译 ChuhanProblem | 10-30 分钟 | ⏳ 进行中 |
| 运行新模拟 | 5-15 分钟 | ⏳ 待编译完成 |
| 数据分析 | 1-2 分钟 | ⏳ 待模拟完成 |
| **总计** | **20-50 分钟** | |

---

## ✨ 总结

✅ 配置文件已修改,时间参数增加了 20 倍  
✅ 编译系统已重新配置为 3D  
⏳ 正在编译新的可执行文件  
🎯 目标: 在 63.2 Myr 的模拟中观察到超新星事件

**期待新的模拟结果能够显示超新星反馈现象!**
