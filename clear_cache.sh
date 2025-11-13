#!/bin/bash
# Clear Python cache to ensure modified code is used
# Run this after making code changes to avoid using stale cached bytecode

echo "Clearing Python cache files..."

# Find and remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ Removed __pycache__ directories"

# Find and remove all .pyc files
echo "Removing .pyc files..."
find /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5 -type f -name "*.pyc" -delete 2>/dev/null
echo "✓ Removed .pyc files"

# Clear JAX cache (if exists)
if [ -d "$HOME/.cache/jax" ]; then
    echo "Clearing JAX compilation cache..."
    rm -rf "$HOME/.cache/jax"
    echo "✓ Cleared JAX cache"
fi

echo ""
echo "Cache clearing complete!"
echo "You can now run your training script."
