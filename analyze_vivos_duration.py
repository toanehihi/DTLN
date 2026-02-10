#!/usr/bin/env python3
"""
Script to analyze audio duration distribution in VIVOS train dataset
"""

import os
import fnmatch
import numpy as np
from wavinfo import WavInfoReader
import matplotlib.pyplot as plt


def analyze_durations(dataset_path):
    """
    Analyze duration distribution of audio files
    
    Args:
        dataset_path: Path to noisy or clean audio directory
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING AUDIO DURATIONS: {dataset_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path not found: {dataset_path}")
        return
    
    # Find all wav files
    wav_files = fnmatch.filter(os.listdir(dataset_path), '*.wav')
    num_files = len(wav_files)
    
    print(f"📊 Found {num_files:,} WAV files")
    print(f"🔍 Analyzing durations...\n")
    
    durations = []
    
    for i, wav_file in enumerate(wav_files):
        file_path = os.path.join(dataset_path, wav_file)
        
        try:
            info = WavInfoReader(file_path)
            duration = info.data.frame_count / info.fmt.sample_rate
            durations.append(duration)
            
            # Progress
            if (i + 1) % 1000 == 0 or (i + 1) == num_files:
                print(f"   Progress: {i+1:,}/{num_files:,} files analyzed", end='\r')
        
        except Exception as e:
            print(f"\n⚠️  Error reading {wav_file}: {e}")
    
    print()  # New line after progress
    
    if not durations:
        print("❌ No valid audio files found!")
        return
    
    # Convert to numpy array
    durations = np.array(durations)
    
    # Statistics
    print(f"\n📊 DURATION STATISTICS:")
    print(f"   Total files:     {len(durations):,}")
    print(f"   Total duration:  {np.sum(durations)/3600:.2f} hours ({np.sum(durations):.2f} seconds)")
    print(f"\n   Min duration:    {np.min(durations):.2f}s")
    print(f"   Max duration:    {np.max(durations):.2f}s")
    print(f"   Mean duration:   {np.mean(durations):.2f}s")
    print(f"   Median duration: {np.median(durations):.2f}s")
    print(f"   Std deviation:   {np.std(durations):.2f}s")
    
    # Percentiles
    print(f"\n📈 PERCENTILES:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(durations, p)
        print(f"   {p:2d}th percentile: {val:.2f}s")
    
    # Sample length analysis
    print(f"\n🎯 SAMPLE LENGTH ANALYSIS:")
    print(f"   (Percentage of files >= sample length)\n")
    
    sample_lengths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    
    for length in sample_lengths:
        count = np.sum(durations >= length)
        percentage = (count / len(durations)) * 100
        bar = '█' * int(percentage / 2)
        print(f"   {length:2d}s: {count:5,} files ({percentage:5.1f}%) {bar}")
    
    # Recommend optimal sample length
    print(f"\n💡 RECOMMENDATIONS:\n")
    
    # Find sample length that keeps 90% of data
    for length in [1, 2, 3, 4, 5, 7, 10, 15]:
        coverage = (np.sum(durations >= length) / len(durations)) * 100
        if coverage >= 90:
            optimal = length
    
    print(f"   ✅ Recommended sample length: {optimal}s")
    print(f"      Keeps ~90%+ of dataset")
    print(f"      Ensures good data utilization")
    
    # Calculate chunks for different sample lengths
    print(f"\n📦 ESTIMATED TRAINING CHUNKS:\n")
    print(f"   Sample   Total      Avg chunks")
    print(f"   Length   Chunks     per file")
    print(f"   " + "-"*35)
    
    for length in [3, 5, 7, 10, 15]:
        total_chunks = 0
        valid_files = 0
        
        for duration in durations:
            chunks = int(duration / length)
            if chunks > 0:
                total_chunks += chunks
                valid_files += 1
        
        avg_chunks = total_chunks / valid_files if valid_files > 0 else 0
        print(f"   {length:2d}s     {total_chunks:6,}     {avg_chunks:.2f}")
    
    # Distribution histogram
    print(f"\n📊 Creating duration histogram...")
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(durations, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of files')
    plt.title(f'Audio Duration Distribution\n{len(durations):,} files')
    plt.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
    plt.axvline(np.median(durations), color='green', linestyle='--', label=f'Median: {np.median(durations):.2f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_durations = np.sort(durations)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    plt.plot(sorted_durations, cumulative, linewidth=2)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Cumulative percentage (%)')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # Mark sample lengths
    for length in [3, 5, 10, 15]:
        idx = np.searchsorted(sorted_durations, length)
        if idx < len(cumulative):
            plt.axvline(length, color='red', linestyle=':', alpha=0.5)
            plt.text(length, cumulative[idx], f'{length}s', rotation=90, va='bottom')
    
    plt.tight_layout()
    plt.savefig('vivos_duration_analysis.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved plot: vivos_duration_analysis.png")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"✅ Analysis complete!")
    print(f"{'='*70}\n")


def main():
    # Analyze train dataset
    train_noisy_path = 'vivos_datasets/train/noisy'
    
    print("\n" + "="*70)
    print("  VIVOS TRAIN DATASET - DURATION ANALYSIS")
    print("="*70)
    
    analyze_durations(train_noisy_path)
    
    print("\n💡 USAGE TIP:")
    print("   Based on the analysis above, update your notebook:")
    print("   ")
    print("   # Section 3: Configuration")
    print("   SAMPLE_LENGTH_SECONDS = <recommended_value>")
    print()


if __name__ == "__main__":
    main()
