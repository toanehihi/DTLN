#!/bin/bash
# Quick start script to create VIVOS + Noise dataset

echo "========================================================================"
echo "  CREATING DTLN DATASET: VIVOS (clean) + DNS Noise"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Clean speech: vivos/"
echo "  - Noise files:  datasets/training_set_sept12/noise/"
echo "  - Output:       vivos_datasets/"
echo "  - SNR range:    -5 to 20 dB"
echo "  - Split:        80% train, 10% val, 10% test"
echo ""
echo "========================================================================"
echo ""

# Check if required directories exist
if [ ! -d "vivos" ]; then
    echo "❌ Error: vivos/ directory not found!"
    echo "   Please make sure VIVOS dataset is in the current directory."
    exit 1
fi

if [ ! -d "datasets/training_set_sept12/noise" ]; then
    echo "❌ Error: datasets/training_set_sept12/noise/ not found!"
    echo "   Please check the noise directory path."
    exit 1
fi

# Count files
echo "📊 Dataset information:"
vivos_train=$(find vivos/train/waves -name "*.wav" 2>/dev/null | wc -l)
vivos_test=$(find vivos/test/waves -name "*.wav" 2>/dev/null | wc -l)
noise_files=$(find datasets/training_set_sept12/noise -name "*.wav" 2>/dev/null | wc -l)

echo "   VIVOS train:    $vivos_train files"
echo "   VIVOS test:     $vivos_test files"
echo "   Total VIVOS:    $((vivos_train + vivos_test)) files"
echo "   Noise files:    $noise_files files"
echo ""

# Ask for confirmation
read -p "Start creating dataset? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting dataset creation..."
echo ""

# Run the Python script with default parameters
python create_dataset_from_dns_vivos.py \
    --vivos-path vivos \
    --dns-noise-path datasets/training_set_sept12/noise \
    --output-path vivos_datasets \
    --sample-rate 16000 \
    --snr-min -5 \
    --snr-max 20 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --workers 8 \
    --seed 42

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Dataset creation completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Output structure:"
    echo "  vivos_datasets/"
    echo "  ├── train/"
    echo "  │   ├── noisy/  (mixed audio)"
    echo "  │   └── clean/  (original clean)"
    echo "  ├── val/"
    echo "  │   ├── noisy/"
    echo "  │   └── clean/"
    echo "  └── test/"
    echo "      ├── noisy/"
    echo "      └── clean/"
    echo ""
    echo "📊 File counts:"
    train_noisy=$(find vivos_datasets/train/noisy -name "*.wav" 2>/dev/null | wc -l)
    val_noisy=$(find vivos_datasets/val/noisy -name "*.wav" 2>/dev/null | wc -l)
    test_noisy=$(find vivos_datasets/test/noisy -name "*.wav" 2>/dev/null | wc -l)
    echo "   Train:      $train_noisy pairs"
    echo "   Validation: $val_noisy pairs"
    echo "   Test:       $test_noisy pairs"
    echo "   Total:      $((train_noisy + val_noisy + test_noisy)) pairs"
    echo ""
    echo "Next steps:"
    echo "  1. Check dataset: python check_dataset.py"
    echo "  2. Train model:   python train.py"
    echo "  3. Or use Colab:  See COLAB_TRAINING_GUIDE.md"
    echo ""
else
    echo ""
    echo "❌ Error: Dataset creation failed!"
    echo "   Please check the error messages above."
    exit 1
fi
