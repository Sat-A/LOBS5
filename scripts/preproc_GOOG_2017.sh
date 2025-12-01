#!/bin/bash
# 预处理 GOOG 2017
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2017/
python /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/preproc.py \
    --data_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_csv/GOOG/2017/ \
    --save_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOG/2017/ \
    --use_raw_book_repr
echo "=== GOOG 2017 DONE ==="
