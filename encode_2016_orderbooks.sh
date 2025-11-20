#!/bin/bash
# 专门处理 2016 年缺失的 orderbook 文件
# 使用现有的 pre_encode_data.py，但只针对 2016 年目录

set -e

INPUT_BASE="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021"
OUTPUT_BASE="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded"

# 创建临时目录，只包含 2016 年的 orderbook 文件
TEMP_INPUT=$(mktemp -d)
TEMP_OUTPUT=$(mktemp -d)

echo "======================================================================="
echo "处理 2016 年缺失的 orderbook 文件"
echo "======================================================================="
echo "输入目录: $INPUT_BASE/2016"
echo "输出目录: $OUTPUT_BASE/2016"
echo "临时目录: $TEMP_INPUT"
echo "======================================================================="

# 创建 2016 子目录
mkdir -p "$TEMP_INPUT/2016"
mkdir -p "$OUTPUT_BASE/2016"

# 只创建缺失 orderbook 文件的符号链接
echo "查找缺失的 orderbook 文件..."
MISSING_COUNT=0
TOTAL_COUNT=0

for orderbook_file in "$INPUT_BASE/2016"/*_orderbook_10_proc.npy; do
    filename=$(basename "$orderbook_file")
    output_file="$OUTPUT_BASE/2016/$filename"
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    # 每 50 个文件显示一次进度
    if [ $((TOTAL_COUNT % 50)) -eq 0 ]; then
        echo "  已检查 $TOTAL_COUNT 个文件，找到 $MISSING_COUNT 个缺失..."
    fi

    # 如果输出文件不存在，创建符号链接到临时目录
    if [ ! -f "$output_file" ]; then
        ln -s "$orderbook_file" "$TEMP_INPUT/2016/$filename"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

echo "检查完成: 总共 $TOTAL_COUNT 个文件，找到 $MISSING_COUNT 个缺失的 orderbook 文件"

if [ $MISSING_COUNT -eq 0 ]; then
    echo "没有缺失的文件！"
    rm -rf "$TEMP_INPUT" "$TEMP_OUTPUT"
    exit 0
fi

# 使用 pre_encode_data.py 处理
# 不使用 skip_files/max_files，所以会处理所有 book 文件
# 由于临时输入目录只包含缺失的 orderbook 文件，所以只会处理这些文件
echo ""
echo "开始使用 pre_encode_data.py 处理..."
echo "命令: python pre_encode_data.py --input_dir $TEMP_INPUT --output_dir $TEMP_OUTPUT --num_workers 1"
echo ""

python pre_encode_data.py \
    --input_dir "$TEMP_INPUT" \
    --output_dir "$TEMP_OUTPUT" \
    --num_workers 1

PYTHON_EXIT_CODE=$?
echo ""
echo "Python 处理完成，退出码: $PYTHON_EXIT_CODE"

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "错误: Python 处理失败！"
    echo "临时目录未删除，请检查: $TEMP_INPUT"
    exit 1
fi

# 将结果复制到最终输出目录
echo ""
echo "复制结果到最终目录..."
echo "从: $TEMP_OUTPUT/2016/"
echo "到: $OUTPUT_BASE/2016/"

# 检查临时输出目录是否有文件
NUM_OUTPUT_FILES=$(ls "$TEMP_OUTPUT/2016"/*_orderbook_10_proc.npy 2>/dev/null | wc -l)
echo "找到 $NUM_OUTPUT_FILES 个处理后的文件"

if [ $NUM_OUTPUT_FILES -gt 0 ]; then
    cp -v "$TEMP_OUTPUT/2016"/*_orderbook_10_proc.npy "$OUTPUT_BASE/2016/"
else
    echo "警告: 没有找到处理后的文件！"
fi

# 清理临时目录
echo ""
echo "清理临时文件..."
rm -rf "$TEMP_INPUT" "$TEMP_OUTPUT"

echo ""
echo "======================================================================="
echo "✓ 完成！"
echo "======================================================================="
