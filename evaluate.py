#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for DTLN Speech Enhancement Model

This script evaluates the trained DTLN model on test dataset and generates:
- Quantitative metrics: SNR, PESQ, STOI, SI-SDR
- Visualizations: Spectrograms, waveforms, comparison plots
- Statistical reports: Mean, std, per-sample results
- Summary report for thesis/project documentation

Author: DTLN Training Project
Date: 2026-02-11
"""

import os
import argparse
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime

# Import DTLN model
from model import DTLN_model

# Try to import evaluation metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("⚠️  PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("⚠️  STOI not available. Install with: pip install pystoi")


def calculate_snr(clean, noisy):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    
    Args:
        clean: Clean audio signal
        noisy: Noisy audio signal
        
    Returns:
        SNR in dB
    """
    noise = noisy - clean
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_si_sdr(reference, estimate):
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    Args:
        reference: Reference clean signal
        estimate: Estimated enhanced signal
        
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Zero-mean normalization
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Calculate SI-SDR
    alpha = np.dot(estimate, reference) / (np.linalg.norm(reference) ** 2 + 1e-8)
    projection = alpha * reference
    noise = estimate - projection
    
    si_sdr = 10 * np.log10(np.sum(projection ** 2) / (np.sum(noise ** 2) + 1e-8))
    return si_sdr


def plot_waveform_comparison(clean, noisy, enhanced, sample_rate, save_path):
    """
    Plot waveform comparison: clean vs noisy vs enhanced
    
    Args:
        clean: Clean audio
        noisy: Noisy audio
        enhanced: Enhanced audio
        sample_rate: Sampling rate
        save_path: Path to save the plot
    """
    duration = len(clean) / sample_rate
    time = np.linspace(0, duration, len(clean))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Clean signal
    axes[0].plot(time, clean, color='green', linewidth=0.5)
    axes[0].set_title('Clean Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])
    
    # Noisy signal
    axes[1].plot(time, noisy, color='red', linewidth=0.5)
    axes[1].set_title('Noisy Signal', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration])
    
    # Enhanced signal
    axes[2].plot(time, enhanced, color='blue', linewidth=0.5)
    axes[2].set_title('Enhanced Signal (DTLN)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrogram_comparison(clean, noisy, enhanced, sample_rate, save_path):
    """
    Plot spectrogram comparison
    
    Args:
        clean: Clean audio
        noisy: Noisy audio
        enhanced: Enhanced audio
        sample_rate: Sampling rate
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Parameters for spectrogram
    n_fft = 512
    hop_length = 128
    
    # Clean spectrogram
    D_clean = np.abs(plt.mlab.specgram(clean, NFFT=n_fft, Fs=sample_rate, 
                                       noverlap=n_fft-hop_length)[0])
    axes[0].imshow(10 * np.log10(D_clean + 1e-10), aspect='auto', origin='lower', 
                   cmap='viridis', extent=[0, len(clean)/sample_rate, 0, sample_rate/2])
    axes[0].set_title('Clean Signal Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_ylim([0, 8000])
    
    # Noisy spectrogram
    D_noisy = np.abs(plt.mlab.specgram(noisy, NFFT=n_fft, Fs=sample_rate,
                                       noverlap=n_fft-hop_length)[0])
    axes[1].imshow(10 * np.log10(D_noisy + 1e-10), aspect='auto', origin='lower',
                   cmap='viridis', extent=[0, len(noisy)/sample_rate, 0, sample_rate/2])
    axes[1].set_title('Noisy Signal Spectrogram', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim([0, 8000])
    
    # Enhanced spectrogram
    D_enhanced = np.abs(plt.mlab.specgram(enhanced, NFFT=n_fft, Fs=sample_rate,
                                          noverlap=n_fft-hop_length)[0])
    im = axes[2].imshow(10 * np.log10(D_enhanced + 1e-10), aspect='auto', origin='lower',
                        cmap='viridis', extent=[0, len(enhanced)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Enhanced Signal Spectrogram (DTLN)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_ylim([0, 8000])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, orientation='vertical', pad=0.01)
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(results_df, save_dir):
    """
    Plot distribution of metrics across test samples
    
    Args:
        results_df: DataFrame with per-sample results
        save_dir: Directory to save plots
    """
    metrics = ['SNR_improvement', 'PESQ_enhanced', 'STOI_enhanced', 'SI-SDR_improvement']
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        data = results_df[metric].dropna()
        
        axes[idx].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {data.mean():.2f}')
        axes[idx].set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def enhance_audio(model, noisy_audio, block_len=512, block_shift=128):
    """
    Enhance audio using DTLN model - OPTIMIZED VERSION
    
    Args:
        model: Trained DTLN model
        noisy_audio: Noisy input audio
        block_len: STFT block length (not used, kept for API compatibility)
        block_shift: STFT hop size (not used, kept for API compatibility)
        
    Returns:
        Enhanced audio
    """
    # Ensure input is 1D
    if len(noisy_audio.shape) > 1:
        noisy_audio = noisy_audio[:, 0]
    
    # DTLN model expects input shape: (batch, time_steps)
    # Add batch dimension
    noisy_batch = np.expand_dims(noisy_audio.astype('float32'), axis=0)
    
    # Process entire audio at once (MUCH faster than block-by-block)
    enhanced_batch = model.predict(noisy_batch, verbose=0)
    
    # Remove batch dimension
    enhanced = enhanced_batch[0]
    
    # Ensure same length as input
    enhanced = enhanced[:len(noisy_audio)]
    
    return enhanced


def evaluate_sample(model, noisy_path, clean_path, sample_rate=16000):
    """
    Evaluate a single test sample
    
    Args:
        model: DTLN model
        noisy_path: Path to noisy audio
        clean_path: Path to clean audio
        sample_rate: Sampling rate
        
    Returns:
        Dictionary with metrics
    """
    # Load audio files
    noisy, sr_noisy = sf.read(noisy_path)
    clean, sr_clean = sf.read(clean_path)
    
    # Ensure same sample rate
    assert sr_noisy == sample_rate and sr_clean == sample_rate, \
        f"Sample rate mismatch: expected {sample_rate}, got {sr_noisy}, {sr_clean}"
    
    # Enhance audio
    enhanced = enhance_audio(model, noisy)
    
    # Ensure same length
    min_len = min(len(clean), len(enhanced), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]
    
    # Calculate metrics
    results = {}
    
    # SNR
    snr_noisy = calculate_snr(clean, noisy)
    snr_enhanced = calculate_snr(clean, enhanced)
    results['SNR_noisy'] = snr_noisy
    results['SNR_enhanced'] = snr_enhanced
    results['SNR_improvement'] = snr_enhanced - snr_noisy
    
    # SI-SDR
    si_sdr_noisy = calculate_si_sdr(clean, noisy)
    si_sdr_enhanced = calculate_si_sdr(clean, enhanced)
    results['SI-SDR_noisy'] = si_sdr_noisy
    results['SI-SDR_enhanced'] = si_sdr_enhanced
    results['SI-SDR_improvement'] = si_sdr_enhanced - si_sdr_noisy
    
    # PESQ (if available)
    if PESQ_AVAILABLE:
        try:
            pesq_noisy = pesq(sample_rate, clean, noisy, 'wb')
            pesq_enhanced = pesq(sample_rate, clean, enhanced, 'wb')
            results['PESQ_noisy'] = pesq_noisy
            results['PESQ_enhanced'] = pesq_enhanced
            results['PESQ_improvement'] = pesq_enhanced - pesq_noisy
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
    
    # STOI (if available)
    if STOI_AVAILABLE:
        try:
            stoi_noisy = stoi(clean, noisy, sample_rate, extended=False)
            stoi_enhanced = stoi(clean, enhanced, sample_rate, extended=False)
            results['STOI_noisy'] = stoi_noisy
            results['STOI_enhanced'] = stoi_enhanced
            results['STOI_improvement'] = stoi_enhanced - stoi_noisy
        except Exception as e:
            print(f"STOI calculation failed: {e}")
    
    return results, clean, noisy, enhanced


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DTLN model on test dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-weights',
        type=str,
        default='models_DTLN_vivos/DTLN_vivos_best.h5',
        help='Path to trained model weights'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='datasets/test',
        help='Path to test dataset directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None = all)'
    )
    
    parser.add_argument(
        '--save-audio',
        action='store_true',
        help='Save enhanced audio files'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots for samples'
    )
    
    parser.add_argument(
        '--max-visualizations',
        type=int,
        default=5,
        help='Maximum number of samples to visualize'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    if args.save_audio:
        os.makedirs(os.path.join(args.output_dir, 'enhanced_audio'), exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    print("=" * 70)
    print("📊 DTLN MODEL EVALUATION")
    print("=" * 70)
    print(f"Model weights: {args.model_weights}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load model
    print("\n🏗️  Loading DTLN model...")
    model_instance = DTLN_model()
    model_instance.build_DTLN_model()
    model_instance.model.load_weights(args.model_weights)
    print("✅ Model loaded successfully!")
    
    # Get test files
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    
    if not os.path.exists(noisy_dir) or not os.path.exists(clean_dir):
        print(f"\n❌ Test directories not found!")
        print(f"Expected: {noisy_dir} and {clean_dir}")
        return 1
    
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
    
    if args.num_samples:
        noisy_files = noisy_files[:args.num_samples]
    
    print(f"\n📁 Found {len(noisy_files)} test samples")
    
    # Evaluate all samples
    print("\n🎯 Evaluating samples...")
    all_results = []
    
    for idx, noisy_file in enumerate(tqdm(noisy_files, desc="Processing")):
        noisy_path = os.path.join(noisy_dir, noisy_file)
        clean_path = os.path.join(clean_dir, noisy_file)
        
        if not os.path.exists(clean_path):
            print(f"⚠️  Clean file not found: {clean_path}")
            continue
        
        try:
            # Evaluate
            results, clean, noisy, enhanced = evaluate_sample(
                model_instance.model, noisy_path, clean_path
            )
            results['filename'] = noisy_file
            all_results.append(results)
            
            # Save enhanced audio
            if args.save_audio:
                enhanced_path = os.path.join(args.output_dir, 'enhanced_audio', noisy_file)
                sf.write(enhanced_path, enhanced, 16000)
            
            # Generate visualizations
            if args.visualize and idx < args.max_visualizations:
                base_name = os.path.splitext(noisy_file)[0]
                
                # Waveform comparison
                waveform_path = os.path.join(args.output_dir, 'visualizations', 
                                              f'{base_name}_waveform.png')
                plot_waveform_comparison(clean, noisy, enhanced, 16000, waveform_path)
                
                # Spectrogram comparison
                spectrogram_path = os.path.join(args.output_dir, 'visualizations',
                                                f'{base_name}_spectrogram.png')
                plot_spectrogram_comparison(clean, noisy, enhanced, 16000, spectrogram_path)
        
        except Exception as e:
            print(f"\n⚠️  Error processing {noisy_file}: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    csv_path = os.path.join(args.output_dir, 'detailed_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed results saved to: {csv_path}")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    metrics_summary = {}
    
    for column in results_df.columns:
        if column != 'filename' and results_df[column].dtype in [np.float64, np.float32]:
            mean_val = results_df[column].mean()
            std_val = results_df[column].std()
            metrics_summary[column] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(results_df[column].min()),
                'max': float(results_df[column].max())
            }
            print(f"{column:25s}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'evaluation_date': datetime.now().isoformat(),
            'model_weights': args.model_weights,
            'num_samples': len(results_df),
            'metrics': metrics_summary
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # Generate metrics distribution plots
    if args.visualize:
        print("\n📊 Generating metric distribution plots...")
        plot_metrics_distribution(results_df, args.output_dir)
        print(f"✅ Saved to: {args.output_dir}/metrics_distribution.png")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {args.output_dir}/")
    print(f"   - Detailed results: detailed_results.csv")
    print(f"   - Summary: summary.json")
    if args.save_audio:
        print(f"   - Enhanced audio: enhanced_audio/")
    if args.visualize:
        print(f"   - Visualizations: visualizations/")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
