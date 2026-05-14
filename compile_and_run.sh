#!/bin/bash
# 编译并运行 ChuhanProblem 的脚本

set -e

echo "=========================================="
echo "ChuhanProblem 编译和运行脚本"
echo "=========================================="
echo ""

BUILD_DIR="/Users/meow/quokka/build"
SOURCE_DIR="/Users/meow/quokka"
TESTS_DIR="$SOURCE_DIR/tests"
CONFIG_FILE="$SOURCE_DIR/inputs/ChuhanProblem.toml"

# 进入构建目录
cd "$BUILD_DIR"

echo "📍 当前目录: $(pwd)"
echo "🔨 正在编译所有目标..."
echo ""

# 编译所有内容
cmake --build . --parallel 4 2>&1 | tail -50

echo ""
echo "✅ 编译完成"
echo ""

# 查找可执行文件
echo "🔍 查找 ChuhanProblem 可执行文件..."
if [ -f "src/problems/ChuhanProblem" ]; then
    EXECUTABLE="src/problems/ChuhanProblem"
    echo "✅ 找到可执行文件: $EXECUTABLE"
elif find . -name "ChuhanProblem" -type f -executable 2>/dev/null | grep -q .; then
    EXECUTABLE=$(find . -name "ChuhanProblem" -type f -executable 2>/dev/null | head -1)
    echo "✅ 找到可执行文件: $EXECUTABLE"
else
    echo "❌ 找不到 ChuhanProblem 可执行文件"
    echo "❌ 编译可能失败。请检查构建日志。"
    exit 1
fi

echo ""
echo "=========================================="
echo "🚀 启动模拟"
echo "=========================================="
echo ""
echo "配置文件: $CONFIG_FILE"
echo "可执行文件: $EXECUTABLE"
echo "工作目录: $TESTS_DIR"
echo ""

# 进入测试目录并运行
cd "$TESTS_DIR"

# 记录开始时间
START=$(date +%s)

# 运行模拟，将输出保存到日志
LOG_FILE="chuhan_run_new_config.log"
echo "运行模拟... (日志: $LOG_FILE)"
echo ""

"$BUILD_DIR/$EXECUTABLE" "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# 记录结束时间
END=$(date +%s)
RUNTIME=$((END - START))

echo ""
echo "=========================================="
echo "✅ 模拟完成！"
echo "=========================================="
echo "运行时间: $RUNTIME 秒 ($(($RUNTIME / 60)) 分 $(($RUNTIME % 60)) 秒)"
echo "输出位置: $TESTS_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 检查输出
if [ -f "chuhan_density_sn_plots/sn_feedback_summary.csv" ]; then
    echo "📊 模拟结果:"
    tail -5 "chuhan_density_sn_plots/sn_feedback_summary.csv"
    echo ""
fi

echo "完成！"
