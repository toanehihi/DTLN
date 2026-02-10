#!/usr/bin/env python3
"""
Script to check dataset statistics including:
- Total size (GB)
- Total duration (hours)
- Number of files
- Average file size and duration
"""

import os
import fnmatch
import soundfile as sf
from wavinfo import WavInfoReader
import numpy as np
from pathlib import Path


def format_size(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_duration(seconds):
    """Convert seconds to hours:minutes:seconds format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def check_dataset_stats(dataset_path, dataset_name="Dataset"):
    """
    Check statistics for a dataset directory
    
    Args:
        dataset_path: Path to noisy or clean audio directory
        dataset_name: Name to display for this dataset
    """
    print(f"\n{'='*60}")
    print(f"Checking {dataset_name}: {dataset_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        return None
    
    # List all .wav files
    file_names = fnmatch.filter(os.listdir(dataset_path), '*.wav')
    num_files = len(file_names)
    
    if num_files == 0:
        print(f"❌ No .wav files found in {dataset_path}")
        return None
    
    print(f"📁 Number of files: {num_files}")
    
    total_size = 0
    total_duration = 0
    total_samples = 0
    sample_rates = set()
    channels = set()
    
    print("🔍 Analyzing files...")
    
    for i, file in enumerate(file_names, 1):
        if i % 100 == 0 or i == num_files:
            print(f"   Progress: {i}/{num_files} files analyzed", end='\r')
        
        file_path = os.path.join(dataset_path, file)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        total_size += file_size
        
        # Get audio info using wavinfo (faster)
        try:
            info = WavInfoReader(file_path)
            duration = info.data.frame_count / info.fmt.sample_rate
            total_duration += duration
            total_samples += info.data.frame_count
            sample_rates.add(info.fmt.sample_rate)
            channels.add(info.fmt.channel_count)
        except Exception as e:
            print(f"\n⚠️  Error reading {file}: {e}")
            continue
    
    print()  # New line after progress
    
    # Calculate statistics
    avg_size = total_size / num_files
    avg_duration = total_duration / num_files
    
    # Display results
    print(f"\n📊 Statistics:")
    print(f"   Total size:        {format_size(total_size)} ({total_size:,} bytes)")
    print(f"   Total duration:    {format_duration(total_duration)} ({total_duration:.2f} seconds)")
    print(f"   Average file size: {format_size(avg_size)}")
    print(f"   Average duration:  {avg_duration:.2f} seconds")
    print(f"   Sample rate(s):    {sorted(sample_rates)} Hz")
    print(f"   Channel(s):        {sorted(channels)}")
    
    return {
        'num_files': num_files,
        'total_size': total_size,
        'total_duration': total_duration,
        'avg_size': avg_size,
        'avg_duration': avg_duration,
        'sample_rates': sample_rates,
        'channels': channels
    }


def main():
    # Define paths
    train_noisy = 'datasets/train/noisy'
    train_clean = 'datasets/train/clean'
    val_noisy = 'datasets/val/noisy'
    val_clean = 'datasets/val/clean'
    test_noisy = 'datasets/test/noisy'
    test_clean = 'datasets/test/clean'
    
    print("\n" + "="*60)
    print("  DTLN DATASET STATISTICS CHECKER")
    print("="*60)
    
    # Check training set
    print("\n" + "🎯 TRAINING SET".center(60))
    train_noisy_stats = check_dataset_stats(train_noisy, "Training Noisy")
    train_clean_stats = check_dataset_stats(train_clean, "Training Clean")
    
    # Check validation set
    print("\n" + "🎯 VALIDATION SET".center(60))
    val_noisy_stats = check_dataset_stats(val_noisy, "Validation Noisy")
    val_clean_stats = check_dataset_stats(val_clean, "Validation Clean")
    
    # Check test set (if exists)
    print("\n" + "🎯 TEST SET".center(60))
    test_noisy_stats = check_dataset_stats(test_noisy, "Test Noisy")
    test_clean_stats = check_dataset_stats(test_clean, "Test Clean")
    
    # Summary
    print("\n" + "="*60)
    print("  OVERALL SUMMARY")
    print("="*60)
    
    total_size = 0
    total_duration = 0
    total_files = 0
    
    for stats in [train_noisy_stats, train_clean_stats, 
                  val_noisy_stats, val_clean_stats,
                  test_noisy_stats, test_clean_stats]:
        if stats:
            total_size += stats['total_size']
            total_duration += stats['total_duration']
            total_files += stats['num_files']
    
    print(f"\n📦 Total dataset:")
    print(f"   Total files:       {total_files:,}")
    print(f"   Total size:        {format_size(total_size)} ({total_size:,} bytes)")
    print(f"   Total duration:    {format_duration(total_duration)} ({total_duration/3600:.2f} hours)")
    
    # Estimate training time
    if train_noisy_stats:
        print(f"\n⏱️  Training estimates:")
        print(f"   Training audio:    {format_duration(train_noisy_stats['total_duration'])} " 
              f"({train_noisy_stats['total_duration']/3600:.2f} hours)")
        print(f"   Training files:    {train_noisy_stats['num_files']:,}")
        
        # Rough estimate: ~2-3 seconds per training step with batch size 64
        # This depends heavily on hardware
        estimated_samples = int(train_noisy_stats['total_duration'] * 16000 / (15 * 128))  # assuming 15s samples, 128 shift
        estimated_batches_per_epoch = estimated_samples // 64  # assuming batch size 64
        estimated_time_per_epoch_min = estimated_batches_per_epoch * 2 / 60  # 2 seconds per batch
        
        print(f"   Estimated samples: ~{estimated_samples:,}")
        print(f"   Est. batches/epoch: ~{estimated_batches_per_epoch:,}")
        print(f"   Est. time/epoch:   ~{estimated_time_per_epoch_min:.1f} min (on GPU)")
        print(f"   Est. 10 epochs:    ~{estimated_time_per_epoch_min*10/60:.1f} hours (on GPU)")
        print(f"   ⚠️  Note: Actual training time depends on your hardware")
    
    print("\n" + "="*60)
    print("✅ Dataset check complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
