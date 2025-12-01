#!/bin/bash
# 验证数据格式
# 用法: bash verify_data.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

python3 << 'EOF'
import numpy as np
from pathlib import Path

def check_file(path, expected_cols, name):
    try:
        data = np.load(path)
        if data.shape[1] == expected_cols:
            print(f"✓ {name}: {data.shape} (correct)")
            return True
        else:
            print(f"✗ {name}: {data.shape} (expected {expected_cols} columns)")
            return False
    except Exception as e:
        print(f"✗ {name}: Error - {e}")
        return False

print("=" * 60)
print("验证预处理数据 (preproc)")
print("=" * 60)

# 检查 preproc 数据
preproc_base = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_preproc")
for ticker in ["GOOG", "GOOGL"]:
    for year in ["2016", "2017", "2018", "2019", "2020", "2021"]:
        msg_files = list((preproc_base / ticker / year).glob("*message*_proc.npy"))
        book_files = list((preproc_base / ticker / year).glob("*orderbook*_proc.npy"))
        if msg_files:
            check_file(msg_files[0], 14, f"{ticker}/{year} message")
        if book_files:
            check_file(book_files[0], 43, f"{ticker}/{year} orderbook")

# JAN2023
jan_preproc = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc")
msg_files = list(jan_preproc.glob("*message*_proc.npy"))
book_files = list(jan_preproc.glob("*orderbook*_proc.npy"))
if msg_files:
    check_file(msg_files[0], 14, "JAN2023 message")
if book_files:
    check_file(book_files[0], 43, "JAN2023 orderbook")

print("\n" + "=" * 60)
print("验证编码数据 (encoded)")
print("=" * 60)

# 检查 encoded 数据
encoded_base = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG_GOOGL_2016TO2021_24tok_encoded")
for ticker in ["GOOG", "GOOGL"]:
    for year in ["2016", "2017", "2018", "2019", "2020", "2021"]:
        msg_files = list((encoded_base / ticker / year).glob("*message*.npy"))
        book_files = list((encoded_base / ticker / year).glob("*orderbook*.npy"))
        if msg_files:
            check_file(msg_files[0], 24, f"{ticker}/{year} encoded msg")
        if book_files:
            check_file(book_files[0], 503, f"{ticker}/{year} encoded book")

# JAN2023 encoded
jan_encoded = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_encoded")
msg_files = list(jan_encoded.glob("*message*.npy"))
book_files = list(jan_encoded.glob("*orderbook*.npy"))
if msg_files:
    check_file(msg_files[0], 24, "JAN2023 encoded msg")
if book_files:
    check_file(book_files[0], 503, "JAN2023 encoded book")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
EOF
