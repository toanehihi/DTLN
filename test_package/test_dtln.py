import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Lambda, Multiply, Activation, Conv1D, Layer,
)

class InstantLayerNormalization(Layer):
    """Per-frame layer normalization (mean/variance over last axis)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-7

    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(
            shape=shape, initializer="ones", trainable=True, name="gamma"
        )
        self.beta = self.add_weight(
            shape=shape, initializer="zeros", trainable=True, name="beta"
        )

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(
            tf.math.square(inputs - mean), axis=[-1], keepdims=True
        )
        std = tf.math.sqrt(variance + self.epsilon)
        return (inputs - mean) / std * self.gamma + self.beta


class DTLNModel:
    """Build, load, and run inference with the DTLN architecture."""

    FS = 16_000
    BLOCK_LEN = 512
    BLOCK_SHIFT = 128
    NUM_UNITS = 128
    NUM_LAYERS = 2
    ENCODER_SIZE = 256
    ACTIVATION = "sigmoid"
    DROPOUT = 0.25

    def __init__(self, weights_path: str, norm_stft: bool = False):
        self.weights_path = weights_path
        self.norm_stft = norm_stft
        self.model = None

    def _stft(self, x):
        frames = tf.signal.frame(x, self.BLOCK_LEN, self.BLOCK_SHIFT)
        stft = tf.signal.rfft(frames)
        return [tf.abs(stft), tf.math.angle(stft)]

    def _ifft(self, x):
        complex_stft = tf.cast(x[0], tf.complex64) * tf.exp(
            1j * tf.cast(x[1], tf.complex64)
        )
        return tf.signal.irfft(complex_stft)

    def _overlap_add(self, x):
        return tf.signal.overlap_and_add(x, self.BLOCK_SHIFT)

    def _separation_kernel(self, mask_size, x):
        for i in range(self.NUM_LAYERS):
            x = LSTM(self.NUM_UNITS, return_sequences=True)(x)
            if i < self.NUM_LAYERS - 1:
                x = Dropout(self.DROPOUT)(x)
        return Activation(self.ACTIVATION)(Dense(mask_size)(x))

    def build(self):
        """Construct the Keras functional model."""
        variant = "norm (10 layers)" if self.norm_stft else "standard (9 layers)"
        print(f"  Building DTLN model [{variant}] ...")

        time_in = Input(batch_shape=(None, None))
        mag, phase = Lambda(self._stft)(time_in)

        mag_input = InstantLayerNormalization()(mag) if self.norm_stft else mag
        mask1 = self._separation_kernel(self.BLOCK_LEN // 2 + 1, mag_input)
        est_frames = Lambda(self._ifft)([Multiply()([mag, mask1]), phase])

        encoded = Conv1D(self.ENCODER_SIZE, 1, use_bias=False)(est_frames)
        encoded_norm = InstantLayerNormalization()(encoded)
        mask2 = self._separation_kernel(self.ENCODER_SIZE, encoded_norm)
        decoded = Conv1D(self.BLOCK_LEN, 1, padding="causal", use_bias=False)(
            Multiply()([encoded, mask2])
        )
        output = Lambda(self._overlap_add)(decoded)

        self.model = Model(inputs=time_in, outputs=output)
        print("  Model built successfully.")

    def load_weights(self):
        """Load .h5 weights into the already-built model."""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        print(f"  Loading weights: {self.weights_path}")
        self.model.load_weights(self.weights_path)
        print("  Weights loaded.")

    def denoise(self, audio: np.ndarray, fs: int) -> np.ndarray:
        """
        Denoise a mono float32 audio array.

        Returns the denoised array with the same length as the input.
        """
        if fs != self.FS:
            audio = scipy_signal.resample(
                audio, int(len(audio) * self.FS / fs)
            )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        original_len = len(audio)

        pad_len = int(np.ceil(original_len / self.BLOCK_SHIFT)) * self.BLOCK_SHIFT
        if pad_len > original_len:
            audio = np.pad(audio, (0, pad_len - original_len))

        result = self.model.predict(
            audio.astype("float32")[np.newaxis], verbose=0
        )[0]
        return result[:original_len]


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a wav file, convert to mono if needed."""
    audio, fs = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, fs


def calculate_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def calculate_snr(clean: np.ndarray, target: np.ndarray) -> float:
    """SNR (dB) of *target* with respect to *clean*."""
    noise = clean - target
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(np.mean(clean ** 2) / noise_power))


LABEL_SIZE = 13
TITLE_SIZE = 14
TICK_SIZE = 11
SUPTITLE_SIZE = 16
COLORS = {"clean": "#1f77b4", "noisy": "#d62728", "denoised": "#2ca02c"}


def _plot_waveform(ax, audio, fs, title, color, ylim):
    t = np.arange(len(audio)) / fs
    ax.plot(t, audio, color=color, linewidth=0.4, alpha=0.85, rasterized=True)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    ax.set_ylabel("Amplitude", fontsize=LABEL_SIZE)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def _plot_spectrogram(ax, audio, fs, title, vmin, vmax):
    f, t, sxx = scipy_signal.spectrogram(audio, fs, nperseg=512, noverlap=384)
    sxx_db = 10 * np.log10(sxx + 1e-10)
    im = ax.pcolormesh(t, f, sxx_db, shading="gouraud", cmap="magma",
                       vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    ax.set_ylabel("Frequency (Hz)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 8000)
    ax.tick_params(labelsize=TICK_SIZE)
    return im


def print_audio_stats(label: str, audio: np.ndarray):
    """Print RMS / Peak / Dynamic range for a signal."""
    rms = calculate_rms(audio)
    peak = float(np.max(np.abs(audio)))
    dynamic = 20 * np.log10(peak / rms) if rms > 0 else 0.0
    print(f"   {label}:")
    print(f"      RMS:      {rms:.6f}")
    print(f"      Peak:     {peak:.6f}")
    print(f"      Dynamic:  {dynamic:.2f} dB")
    return rms


def compare_and_plot(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    fs: int,
    output_plot: str,
):
    """
    Print metrics and generate a clean 2x3 academic figure:
      Row 1  Waveforms   (Clean | Noisy | Denoised)
      Row 2  Spectrograms (Clean | Noisy | Denoised)
    with a concise summary annotation at the bottom.
    """
    min_len = min(len(clean), len(noisy), len(denoised))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    denoised = denoised[:min_len]
    duration = min_len / fs

    print("=" * 70)
    print("  COMPARISON ANALYSIS")
    print("=" * 70)
    print(f"\n  Duration: {duration:.2f}s  |  Samples: {min_len}  |  Fs: {fs} Hz")

    print("\n  AUDIO METRICS")
    print("-" * 70)
    clean_rms = print_audio_stats("Clean", clean)
    noisy_rms = print_audio_stats("Noisy", noisy)
    denoised_rms = print_audio_stats("Denoised", denoised)

    snr_noisy = calculate_snr(clean, noisy)
    snr_denoised = calculate_snr(clean, denoised)
    snr_improvement = snr_denoised - snr_noisy

    print(f"\n  SNR (vs clean reference):")
    print(f"      Noisy:       {snr_noisy:+.2f} dB")
    print(f"      Denoised:    {snr_denoised:+.2f} dB")
    print(f"      Improvement: {snr_improvement:+.2f} dB")

    removed_noise = noisy - denoised
    removed_rms = calculate_rms(removed_noise)
    reduction_db = 20 * np.log10(noisy_rms / (removed_rms + 1e-10))
    print(f"\n  Noise Reduction:")
    print(f"      Removed RMS: {removed_rms:.6f}")
    print(f"      Reduction:   {reduction_db:.2f} dB")

    print("\n  Creating comparison plot ...")

    # Shared axis limits for fair comparison
    amp_max = max(np.max(np.abs(s)) for s in [clean, noisy, denoised])
    amp_lim = (-amp_max * 1.05, amp_max * 1.05)

    all_specs_db = []
    for s in [clean, noisy, denoised]:
        _, _, sxx = scipy_signal.spectrogram(s, fs, nperseg=512, noverlap=384)
        all_specs_db.append(10 * np.log10(sxx + 1e-10))
    spec_vmin = min(db.min() for db in all_specs_db)
    spec_vmax = max(db.max() for db in all_specs_db)

    # Figure: 2 rows x 3 columns
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1, 1, 0.8],
        hspace=0.45, wspace=0.28,
        top=0.91, bottom=0.08, left=0.06, right=0.94,
    )

    fig.suptitle(
        "DTLN Speech Enhancement  —  Waveform & Spectrogram Comparison",
        fontsize=SUPTITLE_SIZE, fontweight="bold", y=0.97,
    )

    axes_wf = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_sp = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_psd = fig.add_subplot(gs[2, :])

    signals = [
        ("(a) Clean",    clean,    COLORS["clean"]),
        ("(b) Noisy",    noisy,    COLORS["noisy"]),
        ("(c) Denoised", denoised, COLORS["denoised"]),
    ]

    for col, (label, sig, color) in enumerate(signals):
        _plot_waveform(axes_wf[col], sig, fs, label, color, amp_lim)
        im = _plot_spectrogram(axes_sp[col], sig, fs, label, spec_vmin, spec_vmax)

    for col in range(3):
        axes_wf[col].set_xlabel("")
        axes_sp[col].set_xlabel("Time (s)", fontsize=LABEL_SIZE)

    for col in range(1, 3):
        axes_wf[col].set_ylabel("")
        axes_sp[col].set_ylabel("")

    cbar = fig.colorbar(im, ax=axes_sp, location="right",
                        fraction=0.018, pad=0.015)
    cbar.set_label("Power (dB)", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    f_clean, pxx_clean = scipy_signal.welch(clean, fs, nperseg=1024)
    f_noisy, pxx_noisy = scipy_signal.welch(noisy, fs, nperseg=1024)
    f_den, pxx_den     = scipy_signal.welch(denoised, fs, nperseg=1024)

    ax_psd.semilogy(f_clean, pxx_clean, color=COLORS["clean"],
                    lw=1.5, alpha=0.85, label="Clean")
    ax_psd.semilogy(f_noisy, pxx_noisy, color=COLORS["noisy"],
                    lw=1.5, alpha=0.7, label="Noisy")
    ax_psd.semilogy(f_den, pxx_den, color=COLORS["denoised"],
                    lw=1.5, alpha=0.85, label="Denoised")
    ax_psd.set_xlim(0, 8000)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=LABEL_SIZE)
    ax_psd.set_ylabel("PSD", fontsize=LABEL_SIZE)
    ax_psd.set_title("(d) Power Spectral Density Comparison",
                     fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    ax_psd.legend(fontsize=LABEL_SIZE, loc="upper right", framealpha=0.9)
    ax_psd.tick_params(labelsize=TICK_SIZE)
    ax_psd.grid(True, alpha=0.2, linewidth=0.5)

    # Summary annotation
    summary = (
        f"SNR input: {snr_noisy:+.1f} dB   |   "
        f"SNR output: {snr_denoised:+.1f} dB   |   "
        f"SNR improvement: {snr_improvement:+.1f} dB   |   "
        f"Noise reduction: {reduction_db:.1f} dB"
    )
    fig.text(
        0.50, 0.02, summary,
        ha="center", va="center", fontsize=12, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0",
                  edgecolor="#cccccc", alpha=0.9),
    )

    plt.savefig(output_plot, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Plot saved: {output_plot}")
    plt.show()

    if snr_improvement > 5:
        quality = "Excellent"
    elif snr_improvement > 2:
        quality = "Good"
    elif snr_improvement > 0:
        quality = "Moderate"
    else:
        quality = "Poor"

    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"  SNR improvement: {snr_improvement:+.2f} dB  ->  {quality}")
    print("=" * 70 + "\n")


def main():
    print("=" * 70)
    print("  DTLN  -  Denoise & Compare Pipeline")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    weights_path = os.path.join(script_dir, "model", "DTLN_vivos_best.h5")
    noisy_path   = os.path.join(script_dir, "test_audio", "noisy_sample.wav")
    clean_path   = os.path.join(script_dir, "test_audio", "clean_sample.wav")
    output_wav   = os.path.join(script_dir, "test_audio", "denoised_output.wav")
    output_plot  = os.path.join(script_dir, "test_audio", "comparison_analysis.png")

    for tag, p in [("Weights", weights_path),
                   ("Noisy audio", noisy_path),
                   ("Clean audio", clean_path)]:
        if not os.path.exists(p):
            print(f"\n  ERROR: {tag} not found -> {p}")
            sys.exit(1)

    use_norm = "norm" in os.path.basename(weights_path).lower()
    print(f"\n  Model : {os.path.basename(weights_path)}")
    print(f"  Variant: {'norm' if use_norm else 'standard'}\n")

    dtln = DTLNModel(weights_path, norm_stft=use_norm)
    dtln.build()
    dtln.load_weights()

    print("\n  Loading audio ...")
    noisy, fs_noisy = load_audio(noisy_path)
    clean, fs_clean = load_audio(clean_path)
    print(f"    Noisy : {noisy_path}  ({len(noisy)/fs_noisy:.2f}s)")
    print(f"    Clean : {clean_path}  ({len(clean)/fs_clean:.2f}s)")

    print("\n  Running inference ...")
    denoised = dtln.denoise(noisy, fs_noisy)

    sf.write(output_wav, denoised, DTLNModel.FS)
    print(f"  Denoised audio saved: {output_wav}")

    print()
    compare_and_plot(clean, noisy, denoised, DTLNModel.FS, output_plot)

    print(f"  Output audio : {output_wav}")
    print(f"  Output plot  : {output_plot}\n")


if __name__ == "__main__":
    main()
