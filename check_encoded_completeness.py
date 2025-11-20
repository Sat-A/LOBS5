#!/usr/bin/env python3
"""
全面检查编码数据的完整性和正确性

功能：
1. 对比原始和编码目录的文件数量
2. 检查缺失文件
3. 采样验证数据格式
4. 检查文件大小是否合理
5. 验证数据内容正确性
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random


def compare_file_counts(original_dir: Path, encoded_dir: Path) -> Dict:
    """对比文件数量"""
    print("=" * 70)
    print("[1] 文件数量对比")
    print("=" * 70)

    # 统计原始文件
    orig_msg = sorted(list(original_dir.glob("**/*message*.npy")))
    orig_book = sorted(list(original_dir.glob("**/*orderbook*.npy")))

    # 统计编码文件
    enc_msg = sorted(list(encoded_dir.glob("**/*message*.npy")))
    enc_book = sorted(list(encoded_dir.glob("**/*orderbook*.npy")))

    results = {
        'original_messages': len(orig_msg),
        'encoded_messages': len(enc_msg),
        'original_orderbooks': len(orig_book),
        'encoded_orderbooks': len(enc_book),
        'orig_msg_files': orig_msg,
        'enc_msg_files': enc_msg,
        'orig_book_files': orig_book,
        'enc_book_files': enc_book
    }

    print(f"Message 文件:")
    print(f"  原始目录: {results['original_messages']}")
    print(f"  编码目录: {results['encoded_messages']}")
    if results['original_messages'] == results['encoded_messages']:
        print(f"  ✓ 数量匹配")
    else:
        print(f"  ✗ 数量不匹配！差异: {results['original_messages'] - results['encoded_messages']}")

    print(f"\nOrderbook 文件:")
    print(f"  原始目录: {results['original_orderbooks']}")
    print(f"  编码目录: {results['encoded_orderbooks']}")
    if results['original_orderbooks'] == results['encoded_orderbooks']:
        print(f"  ✓ 数量匹配")
    else:
        print(f"  ✗ 数量不匹配！差异: {results['original_orderbooks'] - results['encoded_orderbooks']}")

    return results


def find_missing_files(original_dir: Path, encoded_dir: Path, file_type: str) -> List[Path]:
    """查找缺失的文件"""
    pattern = f"**/*{file_type}*.npy"
    orig_files = {f.relative_to(original_dir): f for f in original_dir.glob(pattern)}
    enc_files = {f.relative_to(encoded_dir) for f in encoded_dir.glob(pattern)}

    missing = []
    for rel_path in orig_files.keys():
        if rel_path not in enc_files:
            missing.append(orig_files[rel_path])

    return missing


def check_missing_files(original_dir: Path, encoded_dir: Path) -> bool:
    """检查缺失文件"""
    print("\n" + "=" * 70)
    print("[2] 检查缺失文件")
    print("=" * 70)

    missing_msg = find_missing_files(original_dir, encoded_dir, "message")
    missing_book = find_missing_files(original_dir, encoded_dir, "orderbook")

    all_complete = True

    if missing_msg:
        print(f"\n✗ 缺失 {len(missing_msg)} 个 message 文件:")
        for f in missing_msg[:10]:
            print(f"  - {f.relative_to(original_dir)}")
        if len(missing_msg) > 10:
            print(f"  ... 还有 {len(missing_msg) - 10} 个")
        all_complete = False
    else:
        print("✓ 所有 message 文件都已编码")

    if missing_book:
        print(f"\n✗ 缺失 {len(missing_book)} 个 orderbook 文件:")
        for f in missing_book[:10]:
            print(f"  - {f.relative_to(original_dir)}")
        if len(missing_book) > 10:
            print(f"  ... 还有 {len(missing_book) - 10} 个")
        all_complete = False
    else:
        print("✓ 所有 orderbook 文件都已转换")

    return all_complete


def check_file_formats(encoded_dir: Path, num_samples: int = 10) -> bool:
    """采样检查文件格式"""
    print("\n" + "=" * 70)
    print(f"[3] 采样检查文件格式 (抽取 {num_samples} 个样本)")
    print("=" * 70)

    msg_files = list(encoded_dir.glob("**/*message*.npy"))
    book_files = list(encoded_dir.glob("**/*orderbook*.npy"))

    # 随机抽样
    msg_samples = random.sample(msg_files, min(num_samples, len(msg_files)))
    book_samples = random.sample(book_files, min(num_samples, len(book_files)))

    all_valid = True

    # 检查 message 文件
    print("\nMessage 文件格式检查:")
    for i, msg_file in enumerate(msg_samples, 1):
        try:
            data = np.load(msg_file)
            if data.ndim != 2 or data.shape[1] != 22 or data.dtype != np.int32:
                print(f"  ✗ 样本 {i}: {msg_file.name}")
                print(f"     期望: (N, 22) int32, 实际: {data.shape} {data.dtype}")
                all_valid = False
            else:
                print(f"  ✓ 样本 {i}: {msg_file.name} - {data.shape} {data.dtype}")
        except Exception as e:
            print(f"  ✗ 样本 {i}: {msg_file.name} - 错误: {e}")
            all_valid = False

    # 检查 orderbook 文件
    print("\nOrderbook 文件格式检查:")
    for i, book_file in enumerate(book_samples, 1):
        try:
            data = np.load(book_file)
            if data.ndim != 2 or data.shape[1] != 503 or data.dtype != np.float32:
                print(f"  ✗ 样本 {i}: {book_file.name}")
                print(f"     期望: (N, 503) float32, 实际: {data.shape} {data.dtype}")
                all_valid = False
            else:
                print(f"  ✓ 样本 {i}: {book_file.name} - {data.shape} {data.dtype}")
        except Exception as e:
            print(f"  ✗ 样本 {i}: {book_file.name} - 错误: {e}")
            all_valid = False

    return all_valid


def check_file_sizes(original_dir: Path, encoded_dir: Path, num_samples: int = 5) -> bool:
    """检查文件大小是否合理"""
    print("\n" + "=" * 70)
    print(f"[4] 文件大小合理性检查 (抽取 {num_samples} 个样本)")
    print("=" * 70)

    msg_files = list(original_dir.glob("**/*message*.npy"))
    msg_samples = random.sample(msg_files, min(num_samples, len(msg_files)))

    print("\nMessage 文件大小对比:")
    all_reasonable = True

    for orig_file in msg_samples:
        rel_path = orig_file.relative_to(original_dir)
        enc_file = encoded_dir / rel_path

        if not enc_file.exists():
            print(f"  ✗ 编码文件不存在: {rel_path}")
            all_reasonable = False
            continue

        orig_size = orig_file.stat().st_size / (1024**2)  # MB
        enc_size = enc_file.stat().st_size / (1024**2)    # MB
        ratio = enc_size / orig_size if orig_size > 0 else 0

        # Message 编码后应该变小（14 int64 -> 22 int32）
        # 期望比例：(22 * 4) / (14 * 8) = 88 / 112 = 0.786
        if 0.5 < ratio < 1.2:
            print(f"  ✓ {rel_path.name}")
            print(f"     原始: {orig_size:.1f}MB, 编码: {enc_size:.1f}MB, 比例: {ratio:.2f}")
        else:
            print(f"  ⚠ {rel_path.name} - 大小比例异常")
            print(f"     原始: {orig_size:.1f}MB, 编码: {enc_size:.1f}MB, 比例: {ratio:.2f}")

    print("\nOrderbook 文件大小对比:")
    book_files = list(original_dir.glob("**/*orderbook*.npy"))
    book_samples = random.sample(book_files, min(num_samples, len(book_files)))

    for orig_file in book_samples:
        rel_path = orig_file.relative_to(original_dir)
        enc_file = encoded_dir / rel_path

        if not enc_file.exists():
            print(f"  ✗ 编码文件不存在: {rel_path}")
            all_reasonable = False
            continue

        orig_size = orig_file.stat().st_size / (1024**2)  # MB
        enc_size = enc_file.stat().st_size / (1024**2)    # MB
        ratio = enc_size / orig_size if orig_size > 0 else 0

        # Orderbook 转换后可能增大也可能减小，取决于原始列数
        print(f"  · {rel_path.name}")
        print(f"     原始: {orig_size:.1f}MB, 转换: {enc_size:.1f}MB, 比例: {ratio:.2f}")

    return all_reasonable


def verify_data_correctness(original_dir: Path, encoded_dir: Path, num_samples: int = 3) -> bool:
    """验证数据内容正确性（与 verify_encoded_data.py 中的 compare_with_original 类似）"""
    print("\n" + "=" * 70)
    print(f"[5] 数据内容正确性验证 (抽取 {num_samples} 个样本)")
    print("=" * 70)

    from lob.encoding import Vocab, encode_msgs
    from preproc import transform_L2_state
    import jax.numpy as jnp

    # Initialize vocab
    vocab = Vocab()

    msg_files = list(original_dir.glob("**/*message*.npy"))
    msg_samples = random.sample(msg_files, min(num_samples, len(msg_files)))

    all_correct = True

    print("\nMessage 编码验证:")
    for orig_file in msg_samples:
        rel_path = orig_file.relative_to(original_dir)
        enc_file = encoded_dir / rel_path

        if not enc_file.exists():
            print(f"  ✗ {rel_path.name} - 编码文件不存在")
            all_correct = False
            continue

        try:
            # 加载并重新编码
            X_raw = np.load(orig_file)
            X_expected = encode_msgs(jnp.array(X_raw), vocab.ENCODING)
            X_expected = np.array(X_expected, dtype=np.int32)

            # 加载编码文件
            X_actual = np.load(enc_file)

            # 对比
            if np.array_equal(X_expected, X_actual):
                print(f"  ✓ {rel_path.name} - 编码正确")
            else:
                print(f"  ✗ {rel_path.name} - 编码不匹配!")
                print(f"     期望 shape: {X_expected.shape}, 实际 shape: {X_actual.shape}")
                all_correct = False
        except Exception as e:
            print(f"  ✗ {rel_path.name} - 验证错误: {e}")
            all_correct = False

    print("\nOrderbook 转换验证:")
    book_files = list(original_dir.glob("**/*orderbook*.npy"))
    book_samples = random.sample(book_files, min(num_samples, len(book_files)))

    for orig_file in book_samples:
        rel_path = orig_file.relative_to(original_dir)
        enc_file = encoded_dir / rel_path

        if not enc_file.exists():
            print(f"  ✗ {rel_path.name} - 转换文件不存在")
            all_correct = False
            continue

        try:
            # 加载并重新转换
            book_raw = np.load(orig_file)
            book_expected = transform_L2_state(jnp.array(book_raw), 500, 100)
            book_expected = np.array(book_expected, dtype=np.float32)

            # 加载转换文件
            book_actual = np.load(enc_file)

            # 对比（浮点数用 allclose）
            if np.allclose(book_expected, book_actual, rtol=1e-5, atol=1e-6):
                print(f"  ✓ {rel_path.name} - 转换正确")
            else:
                print(f"  ✗ {rel_path.name} - 转换不匹配!")
                print(f"     期望 shape: {book_expected.shape}, 实际 shape: {book_actual.shape}")
                max_diff = np.abs(book_expected - book_actual).max()
                print(f"     最大差异: {max_diff}")
                all_correct = False
        except Exception as e:
            print(f"  ✗ {rel_path.name} - 验证错误: {e}")
            all_correct = False

    return all_correct


def check_year_completeness(original_dir: Path, encoded_dir: Path) -> bool:
    """检查每年的数据完整性"""
    print("\n" + "=" * 70)
    print("[6] 按年份检查完整性")
    print("=" * 70)

    years = sorted([d.name for d in original_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    all_complete = True
    for year in years:
        orig_year_dir = original_dir / year
        enc_year_dir = encoded_dir / year

        if not enc_year_dir.exists():
            print(f"\n✗ {year}: 编码目录不存在")
            all_complete = False
            continue

        orig_msg = len(list(orig_year_dir.glob("*message*.npy")))
        enc_msg = len(list(enc_year_dir.glob("*message*.npy")))
        orig_book = len(list(orig_year_dir.glob("*orderbook*.npy")))
        enc_book = len(list(enc_year_dir.glob("*orderbook*.npy")))

        status_msg = "✓" if orig_msg == enc_msg else "✗"
        status_book = "✓" if orig_book == enc_book else "✗"

        print(f"\n{year}:")
        print(f"  Message:   {status_msg} {enc_msg}/{orig_msg}")
        print(f"  Orderbook: {status_book} {enc_book}/{orig_book}")

        if orig_msg != enc_msg or orig_book != enc_book:
            all_complete = False

    return all_complete


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="全面检查编码数据的完整性和正确性"
    )

    parser.add_argument(
        "--original",
        type=str,
        default="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021",
        help="原始数据目录"
    )

    parser.add_argument(
        "--encoded",
        type=str,
        default="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded",
        help="编码数据目录"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="每项检查的采样数量"
    )

    args = parser.parse_args()

    original_dir = Path(args.original)
    encoded_dir = Path(args.encoded)

    if not original_dir.exists():
        print(f"错误: 原始目录不存在: {original_dir}")
        sys.exit(1)

    if not encoded_dir.exists():
        print(f"错误: 编码目录不存在: {encoded_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("编码数据完整性检查")
    print("=" * 70)
    print(f"原始目录: {original_dir}")
    print(f"编码目录: {encoded_dir}")
    print("=" * 70)

    # 执行所有检查
    results = {}

    results['counts'] = compare_file_counts(original_dir, encoded_dir)
    results['missing'] = check_missing_files(original_dir, encoded_dir)
    results['formats'] = check_file_formats(encoded_dir, args.samples)
    results['sizes'] = check_file_sizes(original_dir, encoded_dir, args.samples)
    results['years'] = check_year_completeness(original_dir, encoded_dir)
    results['correctness'] = verify_data_correctness(original_dir, encoded_dir, min(args.samples, 3))

    # 最终总结
    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)

    all_pass = all([
        results['missing'],
        results['formats'],
        results['sizes'],
        results['years'],
        results['correctness']
    ])

    if all_pass:
        print("✅ 所有检查通过！数据编码完整且正确")
        print(f"   - Message 文件: {results['counts']['encoded_messages']}")
        print(f"   - Orderbook 文件: {results['counts']['encoded_orderbooks']}")
        print(f"   - 总计: {results['counts']['encoded_messages'] + results['counts']['encoded_orderbooks']} 个文件")
    else:
        print("❌ 发现问题，请检查上述错误")
        if not results['missing']:
            print("   - 有文件缺失")
        if not results['formats']:
            print("   - 文件格式有误")
        if not results['years']:
            print("   - 某些年份数据不完整")
        if not results['correctness']:
            print("   - 数据内容验证失败")

    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
