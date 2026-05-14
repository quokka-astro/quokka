#!/bin/bash
# 运行修改后的 ChuhanProblem 模拟脚本

echo "=================================="
echo "Quokka ChuhanProblem 模拟运行脚本"
echo "=================================="
echo ""

# 检查构建目录
BUILD_DIR="/Users/meow/quokka/build"
SOURCE_DIR="/Users/meow/quokka"
EXECUTABLE="$BUILD_DIR/src/problems/ChuhanProblem"
CONFIG_FILE="$SOURCE_DIR/inputs/ChuhanProblem.toml"

echo "📋 环境检查："
echo "  构建目录: $BUILD_DIR"
echo "  可执行文件: $EXECUTABLE"
echo "  配置文件: $CONFIG_FILE"
echo ""

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "⚠️  可执行文件不存在，正在编译..."
    cd "$BUILD_DIR"
    cmake --build . --target ChuhanProblem
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败！"
        exit 1
    fi
fi

echo "✅ 可执行文件已就绪"
echo ""

# 显示时间参数
echo "⏱️  模拟时间参数："
echo "  初始时间步长 (initial_dt): 2.0e13 秒 ≈ 0.632 Myr"
echo "  最大时间步数 (max_timesteps): 100 步"
echo "  总模拟时间: 100 × 2.0e13 ≈ 63.2 Myr"
echo "  预期输出数量: ~100 个 plotfile"
echo ""

echo "📊 物理参数 (保持不变):"
echo "  环境密度: 1.0e5 cm⁻³"
echo "  环境温度: 10 K"
echo "  化学反馈: 启用 (SNII, WR, AGB)"
echo ""

# 备份旧的输出目录（可选）
TESTS_DIR="$SOURCE_DIR/tests"
BACKUP_DIR="$TESTS_DIR/chuhan_run_backup_$(date +%Y%m%d_%H%M%S)"

if [ -d "$TESTS_DIR/chuhan_density_sn_plots" ]; then
    echo "💾 备份旧的输出数据..."
    mkdir -p "$BACKUP_DIR"
    cp -r "$TESTS_DIR/chuhan_density_sn_plots" "$BACKUP_DIR/" 2>/dev/null || true
    cp -r "$TESTS_DIR/chuhan_metallicity_plots" "$BACKUP_DIR/" 2>/dev/null || true
    echo "  备份位置: $BACKUP_DIR"
    echo ""
fi

# 运行模拟
echo "🚀 启动模拟..."
echo "===================="
cd "$TESTS_DIR"

# 开始计时
START_TIME=$(date +%s)

# 运行程序并将输出重定向到日志文件
LOG_FILE="chuhan_sn_run_extended.log"
"$EXECUTABLE" "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# 结束计时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "===================="
echo "✅ 模拟完成！"
echo ""

# 显示统计信息
echo "📊 模拟统计:"
echo "  运行时间: $((ELAPSED / 60)) 分钟 $((ELAPSED % 60)) 秒"
echo "  输出目录: $TESTS_DIR"
echo "  日志文件: $LOG_FILE"
echo ""

# 检查是否生成了 SN
if [ -f "chuhan_density_sn_plots/sn_feedback_summary.csv" ]; then
    echo "📈 超新星事件统计:"
    
    # 获取 SN 事件总数
    SN_TOTAL=$(tail -1 "chuhan_density_sn_plots/sn_feedback_summary.csv" | cut -d',' -f4)
    echo "  总 SN 事件: $SN_TOTAL"
    
    # 获取时间范围
    TIME_MIN=$(head -2 "chuhan_density_sn_plots/sn_feedback_summary.csv" | tail -1 | cut -d',' -f2)
    TIME_MAX=$(tail -1 "chuhan_density_sn_plots/sn_feedback_summary.csv" | cut -d',' -f2)
    echo "  模拟时间范围: $TIME_MIN - $TIME_MAX Myr"
    
    # 查找 SN 事件行
    echo ""
    echo "💣 SN 发生的时刻:"
    grep -n "^[^,]*,[^,]*,[^,]*,[^0]" "chuhan_density_sn_plots/sn_feedback_summary.csv" | head -20 || echo "  未检测到 SN 事件"
    
    echo ""
fi

# 显示生成的文件数
echo "📁 生成的文件统计:"
DENSITY_COUNT=$(ls -1 chuhan_density_sn_plots/density_*.png 2>/dev/null | wc -l)
SCALAR_COUNT=$(ls -1 chuhan_metallicity_plots/scalar_*.png 2>/dev/null | wc -l)
PLOTFILE_COUNT=$(ls -1d plt* 2>/dev/null | wc -l)

echo "  密度图: $DENSITY_COUNT 张"
echo "  化学标量图: $SCALAR_COUNT 张"
echo "  Plotfile 目录: $PLOTFILE_COUNT 个"
echo ""

echo "✨ 所有操作完成！"
