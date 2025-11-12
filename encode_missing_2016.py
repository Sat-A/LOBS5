#!/usr/bin/env python3
"""
专门处理 2016 年缺失的 orderbook 文件
"""
import os
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from preproc import transform_L2_state
from tqdm import tqdm

def find_missing_orderbook_files(original_dir, encoded_dir, year='2016'):
    """找到所有缺失的 orderbook 文件"""
    original_year_dir = Path(original_dir) / year
    encoded_year_dir = Path(encoded_dir) / year

    missing_files = []

    # 遍历原始目录中的所有 orderbook 文件
    for orig_file in original_year_dir.glob('*_orderbook_10_proc.npy'):
        encoded_file = encoded_year_dir / orig_file.name
        if not encoded_file.exists():
            missing_files.append(orig_file)

    return sorted(missing_files)

def transform_orderbook_file(input_path, output_path):
    """Transform a single orderbook file using JAX"""
    # 读取原始数据
    book_raw = np.load(input_path)

    # 提取 book_depth 和 tick_size
    book_depth = int(book_raw[0, 1])
    tick_size = int(book_raw[0, 2])

    # 转换为 JAX 数组
    book_raw_jax = jnp.array(book_raw)

    # 使用 JAX 加速的 transform_L2_state
    book_transformed = transform_L2_state(book_raw_jax, book_depth, tick_size)

    # 保存结果
    np.save(output_path, np.array(book_transformed))

def main():
    original_dir = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021"
    encoded_dir = "/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded"

    print("正在查找 2016 年缺失的 orderbook 文件...")
    missing_files = find_missing_orderbook_files(original_dir, encoded_dir, year='2016')

    print(f"\n找到 {len(missing_files)} 个缺失的文件")
    print(f"示例: {missing_files[0].name if missing_files else 'None'}")

    if not missing_files:
        print("没有缺失的文件！")
        return

    # 确保输出目录存在
    encoded_year_dir = Path(encoded_dir) / '2016'
    encoded_year_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始处理 {len(missing_files)} 个文件...")

    success_count = 0
    error_count = 0

    for input_path in tqdm(missing_files, desc="Processing"):
        try:
            output_path = Path(encoded_dir) / '2016' / input_path.name
            transform_orderbook_file(input_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"\n错误处理 {input_path.name}: {e}")
            error_count += 1

    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  总计: {len(missing_files)}")

if __name__ == "__main__":
    main()
