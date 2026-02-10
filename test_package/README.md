# DTLN Model Test Package

Simple test package for DTLN speech enhancement model.

## 📦 Package Structure

```
test_package/
├── model/
│   ├── .gitkeep                        (placeholder)
│   └── [paste DTLN_vivos_best.h5 here]   (paste your trained model here)
├── test_audio/
│   └── noisy_sample.wav               (paste your noisy audio here)
├── test_dtln_inference.py             (inference script)
└── README.md                           (this file)
```

## 🚀 Quick Start

### 1. Setup Package

Create the directories:
```bash
cd test_package
mkdir -p model test_audio
```

### 2. Add Your Files

**Copy trained model:**
```bash
# Copy model (from Colab/Drive or local)
cp /path/to/DTLN_vivos_best.h5 model/

# Or from local training:
cp ../vivos_checkpoints/DTLN_vivos_best.weights.h5 model/
```

**Copy test audio:**
```bash
# Any noisy WAV file:
cp /path/to/noisy_audio.wav test_audio/noisy_sample.wav

# Or from test dataset:
cp ../vivos_datasets/test/noisy/test_000000.wav test_audio/noisy_sample.wav
```

### 3. Run Inference

```bash
python test_dtln_inference.py
```

**Output:**
- Denoised audio saved to: `test_audio/denoised_output.wav`

## 📋 Requirements

```bash
pip install tensorflow soundfile numpy scipy
```

Or if using the project environment:
```bash
pip install -r ../requirements.txt
```

## 🎯 Expected Output

```
======================================================================
  DTLN MODEL INFERENCE TEST
======================================================================
🎯 DTLN Inference Model
   Sample rate: 16000 Hz
   Block length: 512
   Block shift: 128

🏗️  Building DTLN model...
✅ Model built successfully!

📥 Loading weights from: model/DTLN_vivos_best.weights.h5
✅ Weights loaded successfully!

🎵 Processing audio...
   Input:  test_audio/noisy_sample.wav
   Output: test_audio/denoised_output.wav
   Duration: 3.24 seconds (51840 samples)
   Processing with DTLN model...
   ✅ Denoised audio saved!

📊 Statistics:
   Input RMS:  0.045123
   Output RMS: 0.038456
   Max value:  0.234567

======================================================================
✅ INFERENCE COMPLETE!
======================================================================

📁 Output saved to: test_audio/denoised_output.wav
🎧 You can now listen to the denoised audio!
```

## 🎨 Custom Usage

### Process your own audio file:

Edit `test_dtln_inference.py` and change the paths:

```python
# In main() function:
weights_path = "model/DTLN_vivos_best.weights.h5"
input_audio = "test_audio/your_audio.wav"      # Change this
output_audio = "test_audio/your_output.wav"    # Change this
```

### Use as a module:

```python
from test_dtln_inference import DTLNInference

# Create inference model
dtln = DTLNInference("model/DTLN_vivos_best.h5")
dtln.build_model()
dtln.load_weights()

# Process audio
dtln.process_audio(
    "input.wav", 
    "output.wav"
)
```

## 📊 Audio Requirements

- **Format:** WAV (mono or stereo)
- **Sample rate:** Any (will be resampled to 16kHz automatically)
- **Channels:** Mono preferred (stereo will be converted)
- **Length:** Any duration

## ⚙️ Model Details

- **Architecture:** DTLN (Dual-Signal Transformation LSTM Network)
- **Sample rate:** 16 kHz
- **Block size:** 512 samples
- **Hop size:** 128 samples
- **LSTM units:** 128
- **Encoder size:** 256

## 🔧 Troubleshooting

### Model file not found
```
❌ Error: Model weights not found at model/DTLN_vivos_best.h5
```
**Solution:** Ensure your trained model is in `test_package/model/`

### Audio file not found
```
❌ Error: Test audio not found at test_audio/noisy_sample.wav
```
**Solution:** Copy a noisy WAV file to `test_package/test_audio/`

### Sample rate mismatch
```
⚠️  Warning: Sample rate mismatch (48000 Hz). Resampling to 16000 Hz...
```
**This is OK** - audio will be automatically resampled.

## 📝 Notes

- The model works best on speech with background noise
- Input audio is automatically normalized
- Output is saved as 16-bit WAV at 16kHz
- Processing time: ~real-time on modern CPU

## 🎉 Done!

Your denoised audio is ready to use!

For more information, see the main project documentation.
