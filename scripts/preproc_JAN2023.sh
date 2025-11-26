#!/bin/bash
# 预处理 JAN2023 测试数据 (在任意机器运行)
# 用法: bash preproc_JAN2023.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc.py"

echo "=== Processing JAN2023 test data ==="
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/
python "$SCRIPT" \
    --data_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/ \
    --save_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/ \
    --use_raw_book_repr

echo "=== JAN2023 DONE ==="
