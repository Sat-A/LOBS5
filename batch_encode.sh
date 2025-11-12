#!/bin/bash
# Batch encoding script to avoid memory accumulation
# Each batch processes a subset of files, then the process exits to free memory

set -e  # Exit on error

# Configuration
INPUT_DIR="${1:-/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021}"
OUTPUT_DIR="${2:-/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded}"
BATCH_SIZE=35  # Files per batch
NUM_WORKERS=1  # Single worker to avoid multiprocess issues

echo "======================================================================="
echo "Batch Encoding Script"
echo "======================================================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size:       $BATCH_SIZE files per batch"
echo "Workers:          $NUM_WORKERS"
echo "======================================================================="

# Count total message files
TOTAL_FILES=$(find "$INPUT_DIR" -name "*message*.npy" | wc -l)
echo "Total message files: $TOTAL_FILES"

# Calculate number of batches
TOTAL_BATCHES=$(( ($TOTAL_FILES + $BATCH_SIZE - 1) / $BATCH_SIZE ))
echo "Total batches: $TOTAL_BATCHES"
echo "======================================================================="

# Process each batch
for ((BATCH=0; BATCH<$TOTAL_BATCHES; BATCH++)); do
    SKIP=$(( $BATCH * $BATCH_SIZE ))

    echo ""
    echo "=== Batch $(($BATCH + 1))/$TOTAL_BATCHES ==="
    echo "Processing files $SKIP to $(($SKIP + $BATCH_SIZE))..."

    python pre_encode_data.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_workers $NUM_WORKERS \
        --skip_files $SKIP \
        --max_files $BATCH_SIZE

    echo "✓ Batch $(($BATCH + 1))/$TOTAL_BATCHES completed"
    echo "Waiting 2 seconds before next batch..."
    sleep 2
done

echo ""
echo "======================================================================="
echo "✓ All batches completed successfully!"
echo "======================================================================="
