# ChuhanProblem 模拟 - 关键数据速查表

**生成时间**: 2026-04-25 22:52:31

---

## 🎯 超新星(SN)信息 - 快速查看

### 关键数字
```
┌─────────────────────────────────────────┐
│  总超新星事件数          0              │
│  模拟时长              1.5937 Myr      │
│  时间步数              11              │
│  首次SN出现时间        未出现           │
│  最后SN出现时间        未出现           │
└─────────────────────────────────────────┘
```

### 时间序列详情

| 时间步 | 时间 (Myr) | SN数 | 粒子数 | Scalar_0 | Scalar_1 | Scalar_2 |
|--------|-----------|------|---------|----------|----------|----------|
| 0      | 0.0000    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 1      | 0.1000    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 2      | 0.2100    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 3      | 0.3310    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 4      | 0.4641    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 5      | 0.6105    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 6      | 0.7716    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 7      | 0.9487    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 8      | 1.1436    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 9      | 1.3579    | 0    | 0       | 0.0      | 0.0      | 0.0      |
| 10     | 1.5937    | 0    | 0       | 0.0      | 0.0      | 0.0      |

**结论**: 
- ✗ 未检测到任何 SN 事件
- ✗ 化学元素丰度保持为零
- ⚠️ 需要延长模拟时间

---

## 📊 可视化文件统计

### 密度分布图
```
目录: /Users/meow/quokka/tests/chuhan_density_sn_plots/
文件数: 11 张 PNG 图表
时间范围: 0.0 ~ 1.5937 Myr
切片方向: Z 方向 (X-Y 平面投影)

文件列表:
  ✓ density_ts0000000_Slice_z_gasDensity.png  (t=0.0000 Myr)
  ✓ density_ts0000001_Slice_z_gasDensity.png  (t=0.1000 Myr)
  ✓ density_ts0000002_Slice_z_gasDensity.png  (t=0.2100 Myr)
  ✓ density_ts0000003_Slice_z_gasDensity.png  (t=0.3310 Myr)
  ✓ density_ts0000004_Slice_z_gasDensity.png  (t=0.4641 Myr)
  ✓ density_ts0000005_Slice_z_gasDensity.png  (t=0.6105 Myr)
  ✓ density_ts0000006_Slice_z_gasDensity.png  (t=0.7716 Myr)
  ✓ density_ts0000007_Slice_z_gasDensity.png  (t=0.9487 Myr)
  ✓ density_ts0000008_Slice_z_gasDensity.png  (t=1.1436 Myr)
  ✓ density_ts0000009_Slice_z_gasDensity.png  (t=1.3579 Myr)
  ✓ density_ts0000010_Slice_z_gasDensity.png  (t=1.5937 Myr)
```

### 化学标量图
```
目录: /Users/meow/quokka/tests/chuhan_metallicity_plots/
文件数: 9 张 PNG 图表
标量类型: 3 种 (Scalar 0, 1, 2)

文件统计:
  Scalar 0: 3 张 (时间步 0, 5, 10)
  Scalar 1: 3 张 (时间步 0, 5, 10)
  Scalar 2: 3 张 (时间步 0, 5, 10)

样本文件:
  ✓ scalar_0_ts0000000_Slice_z_scalar_0.png
  ✓ scalar_0_ts0000005_Slice_z_scalar_0.png
  ✓ scalar_0_ts0000010_Slice_z_scalar_0.png
  ... (Scalar 1, 2 类似)
```

### SN 分析综合图
```
文件: /Users/meow/quokka/tests/sn_analysis_plots.png
包含: 4 个子图
  1. SN Events vs Time (柱状图)
  2. Cumulative SN Events (折线图)
  3. Chemical Abundance Evolution (多线图)
  4. Particle Detection (折线图)
```

---

## 🔢 模拟参数配置

### 时间参数
```
initial_dt = 2.0e13 秒    (原: 1.0e13 秒)
max_timesteps = 100       (原: 10)

计算:
  单步时间: 2.0e13 s ≈ 0.632 Myr
  总时长: 100 × 2.0e13 s ≈ 63.2 Myr
  实际运行: 11 × 0.1594 Myr ≈ 1.59 Myr (旧数据)
```

### 物理参数
```
环境密度:         1.0e5 cm⁻³
环境温度:         10 K
化学追踪:         启用 (SNII, WR, AGB)
SN 反馈机制:       启用 (SN_thermal_or_thermal_momentum)
网格分辨率:        32³ cells
空间维度:         3D
```

### 输出配置
```
plotfile_interval = 10        (原: 50, 每10步输出)
checkpoint_interval = 10      (原: 50, 每10步保存)
预期 plotfile 数: ~100 个 (修改后)
```

---

## 🎬 动画序列信息

### 密度演化序列
```
时间覆盖: 0.0 ~ 1.5937 Myr
帧数: 11 帧
帧率建议: 2 fps (每帧 0.5 秒)
总时长: ~6 秒的动画

查看命令:
  open /Users/meow/quokka/tests/chuhan_density_sn_plots/density_ts*.png
```

### 化学丰度序列
```
时间覆盖: 0.0 ~ 1.5937 Myr
帧数: 9 帧 (稀疏采样: t=0, 5, 10)
预期: 所有值应为零 (未发生 SN)

对比预期:
  如有 SN: Scalar 值会从 0 跳跃到正值
```

---

## 📈 统计汇总

### 事件计数
```
┌─────────────────────┐
│ 超新星事件:     0   │
│ 粒子检测:       0   │
│ 化学反馈:       0   │
└─────────────────────┘
```

### 时间统计
```
┌────────────────────────────────────┐
│ 最小时间:      0.0000 Myr         │
│ 最大时间:      1.5937 Myr         │
│ 平均时间步长:  0.1594 Myr         │
│ 总覆盖时长:    1.5937 Myr         │
└────────────────────────────────────┘
```

### 化学丰度统计
```
┌──────────────────────────┐
│ Scalar_0 总和:  0.0      │
│ Scalar_1 总和:  0.0      │
│ Scalar_2 总和:  0.0      │
│ 最大值 (全部):  0.0      │
└──────────────────────────┘
```

---

## ✅ 生成文件清单

### 分析结果文件
- [x] `/Users/meow/quokka/CHUHAN_SIMULATION_FULL_REPORT.md` - 详细报告
- [x] `/Users/meow/quokka/EXTENDED_SIMULATION_SUMMARY.md` - 参数修改总结
- [x] `/Users/meow/quokka/SIMULATION_ANALYSIS.md` - 初始分析
- [x] `/Users/meow/quokka/TIME_PARAMETERS_MODIFICATION.md` - 修改记录
- [x] `/Users/meow/quokka/run_full_analysis.py` - 分析脚本

### 可视化文件
- [x] `/Users/meow/quokka/tests/sn_analysis_plots.png` - 综合分析图
- [x] `/Users/meow/quokka/tests/chuhan_density_sn_plots/` (11 张密度图)
- [x] `/Users/meow/quokka/tests/chuhan_metallicity_plots/` (9 张化学图)

### 配置文件
- [x] `/Users/meow/quokka/inputs/ChuhanProblem.toml` - 已修改

---

## 🚀 后续操作建议

### 立即可做
```bash
# 1. 查看所有生成的图表
open /Users/meow/quokka/tests/chuhan_density_sn_plots/
open /Users/meow/quokka/tests/sn_analysis_plots.png

# 2. 查看完整报告
cat /Users/meow/quokka/CHUHAN_SIMULATION_FULL_REPORT.md

# 3. 查看原始数据
cat /Users/meow/quokka/tests/chuhan_density_sn_plots/sn_feedback_summary.csv
```

### 下一步计划
```bash
# 1. 进一步增加时间步数 (需要重新编译和运行)
#    修改 max_timesteps: 100 → 200-500
#    重新运行模拟并分析

# 2. 或调整初始条件
#    增加初始恒星质量
#    修改环境密度参数
```

---

**数据状态**: ✅ 完整  
**分析状态**: ✅ 完成  
**可视化状态**: ✅ 已生成  
**报告状态**: ✅ 已生成  

**关键结论**: 
- SN 事件未发生 (模拟时间不足)
- 气体密度分布正常
- 化学反馈未激活
- **建议**: 增加 max_timesteps 以观察 SN 事件
