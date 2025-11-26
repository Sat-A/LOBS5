#!/bin/bash
# 编码训练数据 (预处理完成后运行)
# 用法: bash encode_training.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/pre_encode_data.py"

echo "=== Encoding training data ==="
python "$SCRIPT" \
    --input_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/ \
    --output_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/ \
    --num_workers=8

echo "=== Training data encoding DONE ==="
