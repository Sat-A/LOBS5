#!/bin/bash
# 编码 GOOGL 2017
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lobs5
mkdir -p /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOGL/2017/
python /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/pre_encode_data.py \
    --input_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc/GOOGL/2017/ \
    --output_dir=/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded/GOOGL/2017/ \
    --num_workers=4
echo "=== GOOGL 2017 ENCODED ==="
