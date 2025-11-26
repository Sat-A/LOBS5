#!/bin/bash
# 预处理 GOOGL 数据 (在机器2运行)
# 用法: bash preproc_GOOGL.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

BASE_CSV="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_csv"
BASE_PREPROC="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc"
SCRIPT="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc.py"

for year in 2016 2017 2018 2019 2020 2021; do
    echo "=== Processing GOOGL $year ==="
    mkdir -p "${BASE_PREPROC}/GOOGL/${year}/"
    python "$SCRIPT" \
        --data_dir="${BASE_CSV}/GOOGL/${year}/" \
        --save_dir="${BASE_PREPROC}/GOOGL/${year}/" \
        --use_raw_book_repr
    echo "=== GOOGL $year DONE ==="
done

echo "=== ALL GOOGL DONE ==="
