#!/usr/bin/env python3
"""
直接处理 2016 年缺失的 orderbook 文件
基于现有的 pre_encode_data.py 逻辑
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # 强制使用 CPU

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import gc

sys.path.insert(0, str(Path(__file__).parent))
from lob.encoding import Vocab
from preproc import transform_L2_state


def transform_orderbook_file(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Transform a single orderbook file"""
    try:
        # 读取原始数据
        book_raw = np.load(input_path)

        # 提取参数
        book_depth = int(book_raw[0, 1])
        tick_size = int(book_raw[0, 2])

        # 转换
        book_raw_jax = jnp.array(book_raw)
        book_transformed = transform_L2_state(book_raw_jax, book_depth, tick_size)

        # 保存
        book_transformed_np = np.asarray(book_transformed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, book_transformed_np)

        # 清理内存
        del book_transformed, book_raw_jax, book_transformed_np, book_raw

        return True, "OK"

    except Exception as e:
        return False, str(e)


def main():
    input_base = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021")
    output_base = Path("/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded")

    year = "2016"
    input_dir = input_base / year
    output_dir = output_base / year

    print("="*70)
    print("处理 2016 年缺失的 orderbook 文件")
    print("="*70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("="*70)

    # 查找缺失的文件
    print("\n查找缺失的文件...")
    missing_files = []

    for input_file in sorted(input_dir.glob("*_orderbook_10_proc.npy")):
        output_file = output_dir / input_file.name
        if not output_file.exists():
            missing_files.append((input_file, output_file))

    print(f"找到 {len(missing_files)} 个缺失的文件")

    if not missing_files:
        print("没有缺失的文件！")
        return

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    print(f"\n开始处理...")
    success_count = 0
    error_count = 0

    for i, (input_path, output_path) in enumerate(tqdm(missing_files, desc="Processing")):
        success, msg = transform_orderbook_file(input_path, output_path)

        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"\n错误: {input_path.name}: {msg}")

        # 每 10 个文件清理一次内存
        if (i + 1) % 10 == 0:
            jax.clear_caches()
            gc.collect()

    # 最终清理
    jax.clear_caches()
    gc.collect()

    print("\n" + "="*70)
    print("完成!")
    print("="*70)
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"总计: {len(missing_files)}")
    print("="*70)


if __name__ == "__main__":
    main()
