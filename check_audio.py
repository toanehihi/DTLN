import soundfile as sf
import os

# Check first few files
files = os.listdir('datasets/train/noisy')[:5]
for f in files:
    path = os.path.join('datasets/train/noisy', f)
    data, sr = sf.read(path)
    duration = len(data) / sr
    print(f"{f}: {duration:.2f}s ({len(data)} samples @ {sr}Hz)")
