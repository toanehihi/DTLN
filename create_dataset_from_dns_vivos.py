#!/usr/bin/env python3
"""
Script to create DTLN training dataset by mixing:
- Clean speech from VIVOS dataset
- Noise from DNS Challenge dataset

Output:
- datasets/train/noisy/*.wav
- datasets/train/clean/*.wav
- datasets/val/noisy/*.wav
- datasets/val/clean/*.wav
- datasets/test/noisy/*.wav
- datasets/test/clean/*.wav
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import random
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import fnmatch


def load_audio(file_path, target_sr=16000):
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
    
    Returns:
        audio: Audio data
        sr: Sample rate
    """
    try:
        audio, sr = sf.read(file_path)
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed (simple method, for better quality use librosa)
        if sr != target_sr:
            # Simple resampling by interpolation
            duration = len(audio) / sr
            new_length = int(duration * target_sr)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
            sr = target_sr
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def calculate_rms(audio):
    """Calculate RMS (Root Mean Square) of audio signal"""
    return np.sqrt(np.mean(audio ** 2))


def mix_audio_with_snr(clean, noise, snr_db):
    """
    Mix clean speech with noise at specified SNR level
    
    Args:
        clean: Clean speech signal
        noise: Noise signal
        snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
        noisy: Mixed noisy signal
    """
    # Ensure noise is at least as long as clean
    if len(noise) < len(clean):
        # Repeat noise if too short
        num_repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, num_repeats)
    
    # Randomly select a segment from noise
    if len(noise) > len(clean):
        start_idx = random.randint(0, len(noise) - len(clean))
        noise = noise[start_idx:start_idx + len(clean)]
    
    # Calculate RMS values
    rms_clean = calculate_rms(clean)
    rms_noise = calculate_rms(noise)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 20)
    scale_factor = rms_clean / (snr_linear * rms_noise + 1e-10)
    
    # Mix signals
    noise_scaled = noise * scale_factor
    noisy = clean + noise_scaled
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.95:
        noisy = noisy * 0.95 / max_val
    
    return noisy


def process_single_file(args):
    """
    Process a single clean speech file with random noise
    
    Args:
        args: Tuple of (clean_file, noise_files, output_dir_noisy, output_dir_clean, 
                       target_sr, snr_range, file_id)
    
    Returns:
        success: Boolean indicating success
    """
    clean_file, noise_files, output_dir_noisy, output_dir_clean, target_sr, snr_range, file_id = args
    
    try:
        # Load clean speech
        clean, sr = load_audio(clean_file, target_sr)
        if clean is None:
            return False
        
        # Normalize clean speech
        clean = normalize_audio(clean)
        
        # Randomly select a noise file
        noise_file = random.choice(noise_files)
        
        # Load noise
        noise, _ = load_audio(noise_file, target_sr)
        if noise is None:
            return False
        
        # Normalize noise
        noise = normalize_audio(noise)
        
        # Random SNR level
        snr_db = random.uniform(snr_range[0], snr_range[1])
        
        # Mix clean with noise
        noisy = mix_audio_with_snr(clean, noise, snr_db)
        
        # Generate output filename
        clean_basename = os.path.basename(clean_file)
        output_filename = f"{file_id}_{clean_basename}"
        
        # Save noisy audio
        noisy_path = os.path.join(output_dir_noisy, output_filename)
        sf.write(noisy_path, noisy, target_sr)
        
        # Save clean audio
        clean_path = os.path.join(output_dir_clean, output_filename)
        sf.write(clean_path, clean, target_sr)
        
        return True
    
    except Exception as e:
        print(f"Error processing {clean_file}: {e}")
        return False


def collect_vivos_files(vivos_path, split='train'):
    """
    Collect all VIVOS audio files for a specific split
    
    Args:
        vivos_path: Path to VIVOS dataset
        split: 'train' or 'test'
    
    Returns:
        files: List of audio file paths
    """
    files = []
    split_path = os.path.join(vivos_path, split, 'waves')
    
    if not os.path.exists(split_path):
        print(f"Warning: VIVOS {split} path not found: {split_path}")
        return files
    
    # Iterate through speaker directories
    for speaker in os.listdir(split_path):
        speaker_path = os.path.join(split_path, speaker)
        if os.path.isdir(speaker_path):
            # Find all .wav files
            wav_files = [
                os.path.join(speaker_path, f)
                for f in os.listdir(speaker_path)
                if f.endswith('.wav')
            ]
            files.extend(wav_files)
    
    return sorted(files)


def collect_dns_noise_files(dns_path):
    """
    Collect all DNS noise files
    
    Args:
        dns_path: Path to DNS noise directory
    
    Returns:
        files: List of noise file paths
    """
    files = []
    
    if not os.path.exists(dns_path):
        print(f"Warning: DNS noise path not found: {dns_path}")
        return files
    
    # Recursively find all .wav files
    for root, dirs, filenames in os.walk(dns_path):
        for filename in fnmatch.filter(filenames, '*.wav'):
            files.append(os.path.join(root, filename))
    
    return sorted(files)


def create_dataset(vivos_path, dns_noise_path, output_base_path, 
                  target_sr=16000, snr_range=(-5, 20),
                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                  num_workers=None):
    """
    Create mixed dataset from VIVOS and DNS noise
    
    Args:
        vivos_path: Path to VIVOS dataset
        dns_noise_path: Path to DNS noise directory
        output_base_path: Base path for output dataset
        target_sr: Target sample rate
        snr_range: Range of SNR values (min, max) in dB
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        num_workers: Number of parallel workers
    """
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print("="*70)
    print("CREATING DTLN DATASET FROM VIVOS + DNS NOISE")
    print("="*70)
    
    # Collect VIVOS files
    print(f"\n📂 Collecting VIVOS files from: {vivos_path}")
    vivos_train_files = collect_vivos_files(vivos_path, 'train')
    vivos_test_files = collect_vivos_files(vivos_path, 'test')
    all_vivos_files = vivos_train_files + vivos_test_files
    
    print(f"   Found {len(vivos_train_files):,} VIVOS train files")
    print(f"   Found {len(vivos_test_files):,} VIVOS test files")
    print(f"   Total: {len(all_vivos_files):,} clean speech files")
    
    if len(all_vivos_files) == 0:
        print("❌ No VIVOS files found! Please check the path.")
        return
    
    # Collect DNS noise files
    print(f"\n📂 Collecting DNS noise files from: {dns_noise_path}")
    noise_files = collect_dns_noise_files(dns_noise_path)
    print(f"   Found {len(noise_files):,} noise files")
    
    if len(noise_files) == 0:
        print("❌ No DNS noise files found! Please check the path.")
        return
    
    # Split dataset
    random.shuffle(all_vivos_files)
    total_files = len(all_vivos_files)
    
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    train_files = all_vivos_files[:train_end]
    val_files = all_vivos_files[train_end:val_end]
    test_files = all_vivos_files[val_end:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(train_files):,} files ({len(train_files)/total_files*100:.1f}%)")
    print(f"   Validation: {len(val_files):,} files ({len(val_files)/total_files*100:.1f}%)")
    print(f"   Test:       {len(test_files):,} files ({len(test_files)/total_files*100:.1f}%)")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Sample rate: {target_sr} Hz")
    print(f"   SNR range:   {snr_range[0]} to {snr_range[1]} dB")
    print(f"   Workers:     {num_workers}")
    
    # Create output directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        if len(files) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} set ({len(files):,} files)")
        print(f"{'='*70}")
        
        # Create directories
        output_dir_noisy = os.path.join(output_base_path, split_name, 'noisy')
        output_dir_clean = os.path.join(output_base_path, split_name, 'clean')
        os.makedirs(output_dir_noisy, exist_ok=True)
        os.makedirs(output_dir_clean, exist_ok=True)
        
        # Prepare arguments for parallel processing
        args_list = [
            (file, noise_files, output_dir_noisy, output_dir_clean, 
             target_sr, snr_range, f"{split_name}_{i:06d}")
            for i, file in enumerate(files)
        ]
        
        # Process files in parallel
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc=f"   Creating {split_name} dataset"
            ))
        
        success_count = sum(results)
        print(f"   ✅ Successfully created {success_count:,}/{len(files):,} file pairs")
        
        if success_count < len(files):
            print(f"   ⚠️  {len(files) - success_count} files failed")
    
    print(f"\n{'='*70}")
    print("✅ Dataset creation complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_base_path}")
    print(f"   Structure:")
    print(f"   {output_base_path}/")
    print(f"   ├── train/")
    print(f"   │   ├── noisy/  ({len(train_files):,} files)")
    print(f"   │   └── clean/  ({len(train_files):,} files)")
    print(f"   ├── val/")
    print(f"   │   ├── noisy/  ({len(val_files):,} files)")
    print(f"   │   └── clean/  ({len(val_files):,} files)")
    print(f"   └── test/")
    print(f"       ├── noisy/  ({len(test_files):,} files)")
    print(f"       └── clean/  ({len(test_files):,} files)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Create DTLN dataset from VIVOS (clean) + DNS (noise)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--vivos-path',
        type=str,
        default='vivos',
        help='Path to VIVOS dataset directory'
    )
    
    parser.add_argument(
        '--dns-noise-path',
        type=str,
        default='datasets/training_set_sept12/noise',
        help='Path to noise directory (default: DNS Challenge noise from training_set_sept12)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='vivos_datasets',
        help='Output directory for created dataset'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Target sample rate in Hz'
    )
    
    parser.add_argument(
        '--snr-min',
        type=float,
        default=-5,
        help='Minimum SNR in dB'
    )
    
    parser.add_argument(
        '--snr-max',
        type=float,
        default=20,
        help='Maximum SNR in dB'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (0-1)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (0-1)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (0-1)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"⚠️  Warning: Train + Val + Test ratios = {total_ratio:.2f} (should be 1.0)")
        print(f"    Normalizing ratios...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Create dataset
    create_dataset(
        vivos_path=args.vivos_path,
        dns_noise_path=args.dns_noise_path,
        output_base_path=args.output_path,
        target_sr=args.sample_rate,
        snr_range=(args.snr_min, args.snr_max),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
