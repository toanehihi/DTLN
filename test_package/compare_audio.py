#!/usr/bin/env python3
"""
Audio Comparison Tool - Compare Noisy vs Denoised Audio

This script visualizes and compares the noisy input with denoised output
to evaluate DTLN model performance.
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal


def calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio in dB"""
    noise = clean - noisy
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_rms(audio):
    """Calculate Root Mean Square"""
    return np.sqrt(np.mean(audio ** 2))


def plot_waveform(ax, audio, fs, title, color='blue'):
    """Plot audio waveform"""
    time = np.arange(len(audio)) / fs
    ax.plot(time, audio, color=color, linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)


def plot_spectrogram(ax, audio, fs, title):
    """Plot spectrogram"""
    f, t, Sxx = signal.spectrogram(audio, fs, nperseg=512, noverlap=384)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    ax.set_ylim(0, 8000)  # Focus on speech range
    
    return im


def compare_audio(noisy_path, denoised_path, output_plot='comparison.png'):
    """
    Compare noisy and denoised audio files
    
    Args:
        noisy_path: Path to noisy audio
        denoised_path: Path to denoised audio
        output_plot: Path to save comparison plot
    """
    
    print("=" * 70)
    print("  AUDIO COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load audio files
    print("\n📥 Loading audio files...")
    noisy, fs_noisy = sf.read(noisy_path)
    denoised, fs_denoised = sf.read(denoised_path)
    
    print(f"   Noisy:    {noisy_path}")
    print(f"   Denoised: {denoised_path}")
    
    # Ensure same length
    min_len = min(len(noisy), len(denoised))
    noisy = noisy[:min_len]
    denoised = denoised[:min_len]
    
    duration = len(noisy) / fs_noisy
    print(f"\n⏱️  Duration: {duration:.2f} seconds ({len(noisy)} samples)")
    
    # Calculate metrics
    print("\n📊 AUDIO METRICS:")
    print("-" * 70)
    
    noisy_rms = calculate_rms(noisy)
    denoised_rms = calculate_rms(denoised)
    
    print(f"   Noisy Audio:")
    print(f"      RMS:        {noisy_rms:.6f}")
    print(f"      Peak:       {np.max(np.abs(noisy)):.6f}")
    print(f"      Dynamic:    {20*np.log10(np.max(np.abs(noisy))/noisy_rms):.2f} dB")
    
    print(f"\n   Denoised Audio:")
    print(f"      RMS:        {denoised_rms:.6f}")
    print(f"      Peak:       {np.max(np.abs(denoised)):.6f}")
    print(f"      Dynamic:    {20*np.log10(np.max(np.abs(denoised))/denoised_rms):.2f} dB")
    
    # Noise reduction estimate
    noise_estimate = noisy - denoised
    noise_rms = calculate_rms(noise_estimate)
    noise_reduction_db = 20 * np.log10(noisy_rms / (noise_rms + 1e-10))
    
    print(f"\n   Noise Reduction:")
    print(f"      Removed noise RMS: {noise_rms:.6f}")
    print(f"      Reduction:         {noise_reduction_db:.2f} dB")
    print(f"      RMS change:        {((denoised_rms/noisy_rms - 1) * 100):.1f}%")
    
    # Spectral analysis
    print("\n🎵 SPECTRAL ANALYSIS:")
    print("-" * 70)
    
    # Calculate energy in different frequency bands
    def energy_in_band(audio, fs, low, high):
        f, Pxx = signal.welch(audio, fs, nperseg=1024)
        mask = (f >= low) & (f <= high)
        return np.sum(Pxx[mask])
    
    bands = [
        ("Low (0-500 Hz)", 0, 500),
        ("Mid (500-2000 Hz)", 500, 2000),
        ("High (2000-8000 Hz)", 2000, 8000)
    ]
    
    for band_name, low, high in bands:
        noisy_energy = energy_in_band(noisy, fs_noisy, low, high)
        denoised_energy = energy_in_band(denoised, fs_denoised, low, high)
        change = ((denoised_energy / noisy_energy - 1) * 100)
        print(f"   {band_name:25s} Change: {change:+6.1f}%")
    
    # Create visualization
    print("\n📈 Creating comparison plots...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Waveforms
    ax1 = plt.subplot(3, 2, 1)
    plot_waveform(ax1, noisy, fs_noisy, '🔊 Noisy Input Waveform', 'red')
    
    ax2 = plt.subplot(3, 2, 2)
    plot_waveform(ax2, denoised, fs_denoised, '✨ Denoised Output Waveform', 'green')
    
    # Spectrograms
    ax3 = plt.subplot(3, 2, 3)
    im3 = plot_spectrogram(ax3, noisy, fs_noisy, '🔊 Noisy Input Spectrogram')
    plt.colorbar(im3, ax=ax3, label='Power (dB)')
    
    ax4 = plt.subplot(3, 2, 4)
    im4 = plot_spectrogram(ax4, denoised, fs_denoised, '✨ Denoised Output Spectrogram')
    plt.colorbar(im4, ax=ax4, label='Power (dB)')
    
    # Difference (estimated noise removed)
    ax5 = plt.subplot(3, 2, 5)
    plot_waveform(ax5, noise_estimate, fs_noisy, '🎚️ Removed Noise (Difference)', 'orange')
    
    # Frequency spectrum comparison
    ax6 = plt.subplot(3, 2, 6)
    f_noisy, Pxx_noisy = signal.welch(noisy, fs_noisy, nperseg=1024)
    f_denoised, Pxx_denoised = signal.welch(denoised, fs_denoised, nperseg=1024)
    
    ax6.semilogy(f_noisy, Pxx_noisy, 'r-', alpha=0.7, label='Noisy', linewidth=2)
    ax6.semilogy(f_denoised, Pxx_denoised, 'g-', alpha=0.7, label='Denoised', linewidth=2)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power Spectral Density')
    ax6.set_title('📊 Frequency Spectrum Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 8000)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved comparison plot: {output_plot}")
    
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    if noise_reduction_db > 5:
        quality = "✅ Excellent"
    elif noise_reduction_db > 2:
        quality = "⭐ Good"
    elif noise_reduction_db > 0:
        quality = "⚠️ Moderate"
    else:
        quality = "❌ Poor"
    
    print(f"\n   Noise Reduction: {noise_reduction_db:.2f} dB - {quality}")
    print(f"   RMS Reduction:   {((1 - denoised_rms/noisy_rms) * 100):.1f}%")
    
    print("\n   Model Performance:")
    if noise_reduction_db > 10:
        print("      🎉 Outstanding! Significant noise reduction achieved.")
    elif noise_reduction_db > 5:
        print("      👍 Very Good! Noticeable improvement in audio quality.")
    elif noise_reduction_db > 2:
        print("      ✓ Good! Moderate noise reduction observed.")
    else:
        print("      ⚠️ Limited noise reduction. Check if model trained correctly.")
    
    print("\n" + "=" * 70)
    print()


def main():
    """Main function"""
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    noisy_path = os.path.join(script_dir, "test_audio/noisy_sample.wav")
    denoised_path = os.path.join(script_dir, "test_audio/denoised_output.wav")
    output_plot = os.path.join(script_dir, "test_audio/comparison_analysis.png")
    
    # Check files exist
    if not os.path.exists(noisy_path):
        print(f"❌ Error: Noisy audio not found: {noisy_path}")
        return
    
    if not os.path.exists(denoised_path):
        print(f"❌ Error: Denoised audio not found: {denoised_path}")
        print("   Please run inference first: python test_dtln_inference.py")
        return
    
    # Compare
    compare_audio(noisy_path, denoised_path, output_plot)


if __name__ == "__main__":
    main()
