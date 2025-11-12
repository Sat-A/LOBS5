#!/usr/bin/env python3
"""
Verify encoded data format and integrity.

This script checks that the encoded message and transformed orderbook files
have the correct shapes, dtypes, and value ranges.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional


def verify_encoded_files(encoded_dir: str, verbose: bool = True) -> bool:
    """
    Verify encoded data files.

    Args:
        encoded_dir: Directory containing encoded data
        verbose: Print detailed information

    Returns:
        True if verification passes, False otherwise
    """
    encoded_path = Path(encoded_dir)

    if not encoded_path.exists():
        print(f"❌ Error: Directory does not exist: {encoded_dir}")
        return False

    # Find files
    message_files = sorted(list(encoded_path.glob("**/*message*.npy")))
    book_files = sorted(list(encoded_path.glob("**/*orderbook*.npy")) +
                       list(encoded_path.glob("**/*book*.npy")))

    if verbose:
        print("=" * 70)
        print("Encoded Data Verification")
        print("=" * 70)
        print(f"Directory: {encoded_dir}")
        print(f"Message files found: {len(message_files)}")
        print(f"Orderbook files found: {len(book_files)}")
        print("=" * 70)

    if not message_files and not book_files:
        print("❌ No encoded files found")
        return False

    all_pass = True

    # Verify message files
    if message_files:
        if verbose:
            print("\n[1] Verifying Message Files...")

        msg = np.load(message_files[0])

        if verbose:
            print(f"    Sample file: {message_files[0].name}")
            print(f"    Shape: {msg.shape}")
            print(f"    Dtype: {msg.dtype}")

        # Check shape
        if msg.ndim != 2:
            print(f"    ❌ Expected 2D array, got {msg.ndim}D")
            all_pass = False
        elif msg.shape[1] != 22:
            print(f"    ❌ Expected 22 tokens per message, got {msg.shape[1]}")
            all_pass = False
        else:
            if verbose:
                print(f"    ✓ Shape correct: {msg.shape[0]:,} messages × 22 tokens")

        # Check dtype
        if msg.dtype != np.int32:
            print(f"    ❌ Expected dtype int32, got {msg.dtype}")
            all_pass = False
        else:
            if verbose:
                print(f"    ✓ Dtype correct: int32")

        # Check value range
        msg_min, msg_max = msg.min(), msg.max()
        if verbose:
            print(f"    Value range: [{msg_min}, {msg_max}]")

        if msg_min < 0 or msg_max > 12012:
            print(f"    ⚠ Warning: Unusual value range (expected [0, 12012])")

        # Check a few more files for consistency
        if verbose and len(message_files) > 1:
            print(f"\n    Checking {min(5, len(message_files))} sample files...")

        for i, msg_file in enumerate(message_files[:5]):
            msg_check = np.load(msg_file)
            if msg_check.shape[1] != 22 or msg_check.dtype != np.int32:
                print(f"    ❌ File {i+1} failed: {msg_file.name}")
                all_pass = False
            elif verbose and i > 0:
                print(f"    ✓ File {i+1}: {msg_check.shape}")

    # Verify orderbook files
    if book_files:
        if verbose:
            print("\n[2] Verifying Orderbook Files...")

        book = np.load(book_files[0])

        if verbose:
            print(f"    Sample file: {book_files[0].name}")
            print(f"    Shape: {book.shape}")
            print(f"    Dtype: {book.dtype}")

        # Check shape
        if book.ndim != 2:
            print(f"    ❌ Expected 2D array, got {book.ndim}D")
            all_pass = False
        elif book.shape[1] != 503:
            print(f"    ❌ Expected 503 features (3 + 500 price levels), got {book.shape[1]}")
            all_pass = False
        else:
            if verbose:
                print(f"    ✓ Shape correct: {book.shape[0]:,} timesteps × 503 features")

        # Check dtype
        if book.dtype != np.float32:
            print(f"    ❌ Expected dtype float32, got {book.dtype}")
            all_pass = False
        else:
            if verbose:
                print(f"    ✓ Dtype correct: float32")

        # Check value range
        book_min, book_max = book.min(), book.max()
        if verbose:
            print(f"    Value range: [{book_min:.2f}, {book_max:.2f}]")

        # Check structure (first 3 columns are delta_p_mid, time_s_norm, time_ns_norm)
        if verbose:
            print(f"    First 3 features (delta_p_mid, time_s_norm, time_ns_norm):")
            print(f"      Column 0 range: [{book[:, 0].min():.2f}, {book[:, 0].max():.2f}]")
            print(f"      Column 1 range: [{book[:, 1].min():.2f}, {book[:, 1].max():.2f}]")
            print(f"      Column 2 range: [{book[:, 2].min():.4f}, {book[:, 2].max():.4f}]")

        # Check a few more files for consistency
        if verbose and len(book_files) > 1:
            print(f"\n    Checking {min(5, len(book_files))} sample files...")

        for i, book_file in enumerate(book_files[:5]):
            book_check = np.load(book_file)
            if book_check.shape[1] != 503 or book_check.dtype != np.float32:
                print(f"    ❌ File {i+1} failed: {book_file.name}")
                all_pass = False
            elif verbose and i > 0:
                print(f"    ✓ File {i+1}: {book_check.shape}")

    # Final summary
    if verbose:
        print("\n" + "=" * 70)
        if all_pass:
            print("✅ Verification PASSED")
            print(f"   - {len(message_files)} message files verified")
            print(f"   - {len(book_files)} orderbook files verified")
        else:
            print("❌ Verification FAILED")
            print("   Please check the errors above")
        print("=" * 70)

    return all_pass


def compare_with_original(
    original_dir: str,
    encoded_dir: str,
    num_samples: int = 3
) -> bool:
    """
    Compare a few samples between original and encoded data.

    Args:
        original_dir: Directory with original preproc data
        encoded_dir: Directory with encoded data
        num_samples: Number of files to compare

    Returns:
        True if samples match, False otherwise
    """
    from lob.encoding import Vocab, encode_msgs
    from preproc import transform_L2_state_numpy
    import jax.numpy as jnp

    original_path = Path(original_dir)
    encoded_path = Path(encoded_dir)

    print("\n" + "=" * 70)
    print("Comparing Original vs Encoded Data")
    print("=" * 70)

    # Initialize vocab
    vocab = Vocab()

    # Get sample files
    orig_msg_files = sorted(list(original_path.glob("**/*message*.npy")))[:num_samples]
    orig_book_files = sorted(list(original_path.glob("**/*orderbook*.npy")))[:num_samples]

    all_match = True

    # Compare message files
    for orig_msg_file in orig_msg_files:
        rel_path = orig_msg_file.relative_to(original_path)
        enc_msg_file = encoded_path / rel_path

        if not enc_msg_file.exists():
            print(f"❌ Missing encoded file: {enc_msg_file}")
            all_match = False
            continue

        print(f"\nChecking: {rel_path}")

        # Load original and encode
        X_raw = np.load(orig_msg_file)
        X_encoded_expected = encode_msgs(jnp.array(X_raw), vocab.ENCODING)
        X_encoded_expected = np.array(X_encoded_expected, dtype=np.int32)

        # Load encoded
        X_encoded_actual = np.load(enc_msg_file)

        # Compare
        if np.array_equal(X_encoded_expected, X_encoded_actual):
            print(f"  ✓ Messages match")
        else:
            print(f"  ❌ Messages differ!")
            print(f"     Expected shape: {X_encoded_expected.shape}")
            print(f"     Actual shape: {X_encoded_actual.shape}")
            all_match = False

    # Compare orderbook files
    for orig_book_file in orig_book_files:
        rel_path = orig_book_file.relative_to(original_path)
        enc_book_file = encoded_path / rel_path

        if not enc_book_file.exists():
            print(f"❌ Missing encoded file: {enc_book_file}")
            all_match = False
            continue

        print(f"\nChecking: {rel_path}")

        # Load original and transform
        book_raw = np.load(orig_book_file)
        book_transformed_expected = transform_L2_state_numpy(book_raw, 500, 100)
        book_transformed_expected = np.array(book_transformed_expected, dtype=np.float32)

        # Load transformed
        book_transformed_actual = np.load(enc_book_file)

        # Compare (use allclose for float comparison)
        if np.allclose(book_transformed_expected, book_transformed_actual, rtol=1e-5):
            print(f"  ✓ Orderbooks match")
        else:
            print(f"  ❌ Orderbooks differ!")
            print(f"     Expected shape: {book_transformed_expected.shape}")
            print(f"     Actual shape: {book_transformed_actual.shape}")
            all_match = False

    print("\n" + "=" * 70)
    if all_match:
        print("✅ All samples match original data")
    else:
        print("❌ Some samples differ from original")
    print("=" * 70)

    return all_match


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify encoded LOBSTER data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "encoded_dir",
        type=str,
        help="Directory containing encoded data"
    )

    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Original directory to compare against"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Verify encoded files
    success = verify_encoded_files(args.encoded_dir, verbose=not args.quiet)

    # Optionally compare with original
    if args.compare and success:
        success = compare_with_original(args.compare, args.encoded_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
