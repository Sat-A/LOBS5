#!/bin/bash
# 预处理 GOOGL 2021
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOGL/2021/
python /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc.py \
    --data_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_csv/GOOGL/2021/ \
    --save_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOGL/2021/ \
    --use_raw_book_repr
echo "=== GOOGL 2021 DONE ==="
