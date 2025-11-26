#!/bin/bash
# 预处理 GOOG 数据 (在机器1运行)
# 用法: bash preproc_GOOG.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

BASE_CSV="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_csv"
BASE_PREPROC="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc"
SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc.py"

for year in 2016 2017 2018 2019 2020 2021; do
    echo "=== Processing GOOG $year ==="
    mkdir -p "${BASE_PREPROC}/GOOG/${year}/"
    python "$SCRIPT" \
        --data_dir="${BASE_CSV}/GOOG/${year}/" \
        --save_dir="${BASE_PREPROC}/GOOG/${year}/" \
        --use_raw_book_repr
    echo "=== GOOG $year DONE ==="
done

echo "=== ALL GOOG DONE ==="
