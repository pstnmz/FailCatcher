#!/bin/bash
# Launcher script for UMAP projection generation across all medMNIST datasets
# Uses raw image features (no model loading required)
# 
# Usage:
#   bash launch_umap_all.sh           # Default: severity 3
#   bash launch_umap_all.sh 5         # Custom severity

set -e  # Exit on error

# Parse arguments
SEVERITY=${1:-3}  # Default corruption severity: 3

# Create output directory
OUTPUT_DIR="./umap_projections_raw_images_sev${SEVERITY}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "UMAP Projection Generation (Raw Images)"
echo "=========================================="
echo "Corruption severity: $SEVERITY"
echo "Using: Flattened raw images"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run UMAP generation
python generate_umap_projections.py \
    --all-datasets \
    --corruption-severity "$SEVERITY" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 512 \
    --gpu 0

echo ""
echo "=========================================="
echo "✓ UMAP projections completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
echo ""
echo "To view embeddings:"
echo "  import numpy as np"
echo "  data = np.load('${OUTPUT_DIR}/umap_organamnist_embeddings.npz')"
echo "  print(data.files)"
