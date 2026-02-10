#!/usr/bin/env python3
"""
Script to check VIVOS dataset statistics including:
- Total size (GB)
- Total duration (hours)
- Number of files
- Number of speakers
- Average file size and duration per speaker
- Transcription analysis
"""

import os
import fnmatch
import soundfile as sf
from wavinfo import WavInfoReader
import numpy as np
from pathlib import Path
from collections import defaultdict


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


def load_prompts(prompts_file):
    """Load transcriptions from prompts.txt file"""
    prompts = {}
    if not os.path.exists(prompts_file):
        return prompts
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    prompts[file_id] = text
    
    return prompts


def check_vivos_dataset(base_path, dataset_type="train"):
    """
    Check statistics for VIVOS dataset (train or test)
    
    Args:
        base_path: Path to vivos directory
        dataset_type: 'train' or 'test'
    """
    dataset_path = os.path.join(base_path, dataset_type)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING VIVOS {dataset_type.upper()} SET: {dataset_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        return None
    
    # Load prompts/transcriptions
    prompts_file = os.path.join(dataset_path, 'prompts.txt')
    prompts = load_prompts(prompts_file)
    print(f"📝 Loaded {len(prompts):,} transcriptions from prompts.txt")
    
    # Find all speaker directories
    waves_path = os.path.join(dataset_path, 'waves')
    if not os.path.exists(waves_path):
        print(f"❌ Waves directory not found: {waves_path}")
        return None
    
    speakers = [d for d in os.listdir(waves_path) if os.path.isdir(os.path.join(waves_path, d))]
    speakers.sort()
    num_speakers = len(speakers)
    
    print(f"🎤 Number of speakers: {num_speakers}")
    print(f"🔍 Analyzing audio files...")
    
    # Statistics
    total_files = 0
    total_size = 0
    total_duration = 0
    total_samples = 0
    sample_rates = set()
    channels = set()
    
    speaker_stats = {}
    missing_audio = []
    missing_transcription = []
    
    # Analyze each speaker
    for speaker_idx, speaker in enumerate(speakers, 1):
        speaker_path = os.path.join(waves_path, speaker)
        
        # Progress indicator
        if speaker_idx % 10 == 0 or speaker_idx == num_speakers:
            print(f"   Progress: {speaker_idx}/{num_speakers} speakers analyzed", end='\r')
        
        # Find all .wav files for this speaker
        wav_files = fnmatch.filter(os.listdir(speaker_path), '*.wav')
        
        speaker_files = 0
        speaker_size = 0
        speaker_duration = 0
        
        for wav_file in wav_files:
            file_path = os.path.join(speaker_path, wav_file)
            file_id = os.path.splitext(wav_file)[0]
            
            # Check if transcription exists
            if file_id not in prompts:
                missing_transcription.append(file_id)
            
            # Get file size
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                speaker_size += file_size
                
                # Get audio info
                info = WavInfoReader(file_path)
                duration = info.data.frame_count / info.fmt.sample_rate
                total_duration += duration
                speaker_duration += duration
                total_samples += info.data.frame_count
                sample_rates.add(info.fmt.sample_rate)
                channels.add(info.fmt.channel_count)
                
                total_files += 1
                speaker_files += 1
                
            except Exception as e:
                print(f"\n⚠️  Error reading {wav_file}: {e}")
                missing_audio.append(file_id)
                continue
        
        # Store speaker stats
        speaker_stats[speaker] = {
            'files': speaker_files,
            'size': speaker_size,
            'duration': speaker_duration
        }
    
    print()  # New line after progress
    
    # Calculate overall statistics
    avg_size = total_size / total_files if total_files > 0 else 0
    avg_duration = total_duration / total_files if total_files > 0 else 0
    
    # Display overall results
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total files:       {total_files:,}")
    print(f"   Total size:        {format_size(total_size)} ({total_size:,} bytes)")
    print(f"   Total duration:    {format_duration(total_duration)} ({total_duration:.2f} seconds / {total_duration/3600:.2f} hours)")
    print(f"   Average file size: {format_size(avg_size)}")
    print(f"   Average duration:  {avg_duration:.2f} seconds")
    print(f"   Sample rate(s):    {sorted(sample_rates)} Hz")
    print(f"   Channel(s):        {sorted(channels)}")
    
    # Speaker statistics
    print(f"\n🎤 SPEAKER STATISTICS:")
    print(f"   Number of speakers: {num_speakers}")
    if speaker_stats:
        avg_files_per_speaker = total_files / num_speakers
        avg_duration_per_speaker = total_duration / num_speakers
        print(f"   Avg files/speaker:  {avg_files_per_speaker:.1f}")
        print(f"   Avg duration/speaker: {format_duration(avg_duration_per_speaker)} ({avg_duration_per_speaker:.2f} seconds)")
    
    # Show top 5 speakers by duration
    if speaker_stats:
        sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['duration'], reverse=True)
        print(f"\n   Top 5 speakers by duration:")
        for i, (speaker, stats) in enumerate(sorted_speakers[:5], 1):
            print(f"      {i}. {speaker}: {stats['files']} files, {format_duration(stats['duration'])}")
    
    # Transcription analysis
    print(f"\n📝 TRANSCRIPTION ANALYSIS:")
    print(f"   Total transcriptions: {len(prompts):,}")
    print(f"   Files with audio:     {total_files:,}")
    print(f"   Match status:         ", end='')
    
    if len(prompts) == total_files:
        print(f"✅ Perfect match!")
    else:
        print(f"⚠️  Mismatch detected")
        if len(missing_transcription) > 0:
            print(f"   Missing transcription for {len(missing_transcription)} files")
        if len(missing_audio) > 0:
            print(f"   Missing/corrupt audio for {len(missing_audio)} files")
    
    # Text analysis
    if prompts:
        all_text = ' '.join(prompts.values())
        total_words = len(all_text.split())
        total_chars = len(all_text)
        avg_words_per_file = total_words / len(prompts)
        
        print(f"\n   Text statistics:")
        print(f"      Total words:      {total_words:,}")
        print(f"      Total characters: {total_chars:,}")
        print(f"      Avg words/file:   {avg_words_per_file:.1f}")
    
    return {
        'num_speakers': num_speakers,
        'num_files': total_files,
        'total_size': total_size,
        'total_duration': total_duration,
        'avg_size': avg_size,
        'avg_duration': avg_duration,
        'sample_rates': sample_rates,
        'channels': channels,
        'speaker_stats': speaker_stats,
        'prompts': prompts
    }


def main():
    # VIVOS dataset path
    vivos_path = 'vivos'
    
    print("\n" + "="*70)
    print("  VIVOS DATASET STATISTICS CHECKER")
    print("  Vietnamese Speech Corpus for ASR")
    print("="*70)
    
    if not os.path.exists(vivos_path):
        print(f"\n❌ VIVOS directory not found: {vivos_path}")
        print("Please make sure the 'vivos' directory exists in the current path.")
        return
    
    # Check training set
    print("\n" + "🎯 TRAINING SET".center(70))
    train_stats = check_vivos_dataset(vivos_path, "train")
    
    # Check test set
    print("\n" + "🎯 TEST SET".center(70))
    test_stats = check_vivos_dataset(vivos_path, "test")
    
    # Overall summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    
    total_speakers = 0
    total_files = 0
    total_size = 0
    total_duration = 0
    
    for stats in [train_stats, test_stats]:
        if stats:
            total_speakers += stats['num_speakers']
            total_files += stats['num_files']
            total_size += stats['total_size']
            total_duration += stats['total_duration']
    
    print(f"\n📦 Complete VIVOS dataset:")
    print(f"   Total speakers:    {total_speakers:,}")
    print(f"   Total files:       {total_files:,}")
    print(f"   Total size:        {format_size(total_size)} ({total_size:,} bytes)")
    print(f"   Total duration:    {format_duration(total_duration)} ({total_duration/3600:.2f} hours)")
    
    # Dataset split ratio
    if train_stats and test_stats:
        train_ratio = train_stats['total_duration'] / total_duration * 100
        test_ratio = test_stats['total_duration'] / total_duration * 100
        
        print(f"\n📊 Dataset split:")
        print(f"   Training:   {train_stats['total_duration']/3600:.2f} hours ({train_ratio:.1f}%)")
        print(f"   Test:       {test_stats['total_duration']/3600:.2f} hours ({test_ratio:.1f}%)")
    
    # Recommendations
    print(f"\n💡 USAGE RECOMMENDATIONS:")
    print(f"   • This is a clean speech corpus for ASR training")
    print(f"   • For DTLN training, you'll need to add noise to create noisy/clean pairs")
    print(f"   • Suggested approach:")
    print(f"      1. Use VIVOS as clean speech")
    print(f"      2. Mix with noise datasets (DNS Challenge, FreeSound, etc.)")
    print(f"      3. Create synthetic noisy audio for speech enhancement")
    print(f"   • Average utterance: {total_duration/total_files if total_files > 0 else 0:.1f}s")
    print(f"     (Good length for model training)")
    
    print("\n" + "="*70)
    print("✅ VIVOS dataset check complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
