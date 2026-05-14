# ChuhanProblem.toml - 时间参数修改记录

**修改时间**: 2026年4月25日

## 修改概览

为了延长模拟时间并观察超新星(SN)事件，对 `ChuhanProblem.toml` 的时间相关参数进行了调整。

---

## 修改内容

### 1. 增加每步时间间隔 (initial_dt)

| 参数 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| `initial_dt` | 1.0e13 秒 | 2.0e13 秒 | ×2 (增加一倍) |

**转换为物理时间:**
- 1.0e13 秒 ≈ **316,000 年** ≈ **0.316 Myr**
- 2.0e13 秒 ≈ **632,000 年** ≈ **0.632 Myr**

### 2. 增加模拟总时间步数 (max_timesteps)

| 参数 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| `max_timesteps` | 10 步 | 100 步 | ×10 (增加10倍) |

### 3. 调整输出间隔

| 参数 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| `plotfile_interval` | 50 | 10 | 更频繁的绘图输出,便于监测SN事件 |
| `checkpoint_interval` | 50 | 10 | 更频繁的检查点保存 |

---

## 模拟时间覆盖范围计算

### 修改前
- **总模拟时间** = max_timesteps × initial_dt
- = 10 × 1.0e13 秒
- ≈ **3.16 Myr** (百万年)

### 修改后
- **总模拟时间** = max_timesteps × initial_dt  
- = 100 × 2.0e13 秒
- ≈ **63.2 Myr** (百万年)

**时间跨度增加: 3.16 → 63.2 Myr (约20倍)**

---

## 预期影响

### ✅ 积极影响
1. **更长的模拟时间**: 从 3.16 Myr 扩展到 63.2 Myr
2. **更大的时间步长**: 从 316k 年/步 增加到 632k 年/步
3. **更多输出数据**: 原来预计 10 个 plotfile，现在预计 100 个
4. **更高的观测概率**: 有更大的时间窗口来捕捉 SN 事件

### ⚠️ 需要注意
1. **计算时间增加**: 模拟会耗时更长 (估计增加20倍左右)
2. **磁盘空间需求**: 输出文件数量增加,需要更多存储空间
3. **数值稳定性**: 更大的时间步长需要验证Courant条件是否满足

---

## SN事件预期

### 恒星演化背景
- 根据IMF和初始条件,恒星应在数百万年内达到生命末期
- SNII(SNe II)通常来自质量 > 8 M☉ 的恒星,寿命 ~ 几百万年
- WR(Wolf-Rayet)阶段可能触发较早的反馈
- AGB(Asymptotic Giant Branch)阶段提供额外的化学丰度

### 观测策略
1. 监测 `plotfile_XXXX/` 目录中的新数据
2. 检查 `sn_feedback_summary.csv` 中 `n_sn_markers` 列是否出现非零值
3. 查看化学标量(scalar)的变化,指示SN注入的金属

---

## 后续建议

### 运行新模拟
```bash
cd /Users/meow/quokka/build
cmake --build . --target ChuhanProblem
./src/problems/ChuhanProblem ../inputs/ChuhanProblem.toml
```

### 监测输出
```bash
# 查看最新的 plotfile
ls -lt tests/plt* | head -10

# 监测 SN 事件
tail -20 tests/chuhan_density_sn_plots/sn_feedback_summary.csv

# 检查化学演化
grep "n_sn_markers\|scalar.*sum" tests/chuhan_sn_run.log | tail -20
```

### 如需进一步调整
- 若仍未出现 SN: 考虑再次增加 `max_timesteps` 至 200-500
- 若SN频繁发生: 考虑使用原始参数或更细的时间步(减小 `initial_dt`)
- 若计算耗时过长: 可以减少 `max_timesteps` 或增加空间分辨率限制

---

## 配置文件位置
- **文件**: `/Users/meow/quokka/inputs/ChuhanProblem.toml`
- **修改者**: AI Assistant
- **修改时间**: 2026年4月25日

---

## 参考参数

**重要物理参数保持不变:**
- 环境密度 (n0): 1.0e5 cm⁻³
- 环境温度 (Tamb): 10 K
- SN方案: `SN_thermal_or_thermal_momentum`
- 化学反馈: 启用 (SNII, WR, AGB)
- 网格分辨率: 32³ cells (单一等级)
