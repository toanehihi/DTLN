# DTLN — Dual-signal Transformation LSTM Network

A lightweight neural network for real-time speech enhancement and noise suppression.

---

## Model Architecture

### Overview

DTLN operates as a two-stage pipeline. Each stage applies a **learned multiplicative mask** to suppress noise while preserving speech content. The key design principle is that the model learns signal representations **directly from raw audio** — no handcrafted feature extraction (e.g., MFCC, mel-filterbanks) is required.

| Stage | Domain | Transform | Purpose |
|-------|--------|-----------|---------|
| 1 | Frequency | STFT | Coarse noise suppression on spectral magnitudes |
| 2 | Learned | 1D Convolution | Fine-grained enhancement on learned representations |

By combining a classical signal-processing transform (STFT) with a data-driven transform (Conv1D), the model achieves robust denoising with only ~2 M parameters — small enough for real-time inference on mobile and edge devices.

### Signal Flow

```
Raw waveform
  │
  ├──► STFT ──► magnitude + phase
  │                │
  │           [LayerNorm]  (optional, norm variant)
  │                │
  │           LSTM × 2 ──► Dense ──► Sigmoid ──► mask₁
  │                │
  │           magnitude × mask₁
  │                │
  │           iFFT (reconstruct time-domain frames)
  │                │
  │           Conv1D Encoder  (512 → 256, kernel=1)
  │                │
  │           Instant Layer Normalization
  │                │
  │           LSTM × 2 ──► Dense ──► Sigmoid ──► mask₂
  │                │
  │           encoded × mask₂
  │                │
  │           Conv1D Decoder  (256 → 512, kernel=1, causal)
  │                │
  └──► Overlap-and-Add ──► Enhanced waveform
```

### Stage 1 — STFT Domain

The input waveform is segmented into overlapping frames (512 samples, hop 128) and transformed via real-valued FFT. The resulting magnitude spectrum is passed through two stacked LSTM layers that predict a sigmoid-bounded mask in the range [0, 1]. Multiplying the original magnitude by this mask attenuates noise-dominated frequency bins while retaining speech harmonics. The masked magnitude and the original phase are then recombined through an inverse FFT to produce time-domain frames.

### Stage 2 — Learned Domain via 1D Convolution

Rather than applying a second mask in the same spectral domain, the model transforms the output of Stage 1 into a **learned feature space** using a pointwise 1D convolutional layer (kernel size = 1). This Conv1D encoder acts as a linear projection from 512-dimensional frame samples to a 256-dimensional latent representation — effectively learning a task-specific transform that captures patterns not easily expressed by the fixed Fourier basis.

After normalization, a second pair of LSTM layers predicts another multiplicative mask over these learned features. A causal Conv1D decoder then projects the masked representation back to the original frame dimension, and overlap-and-add reconstructs the final enhanced waveform.

### Why 1D Convolution Is Effective

Traditional audio processing pipelines rely on hand-designed features (e.g., mel-scale filterbanks, spectral flux). While effective in well-understood scenarios, these fixed representations cannot adapt to the specific noise characteristics present in the training data.

The 1D convolutional layers in DTLN replace this manual step:

- **Data-driven basis functions.** Each filter in the Conv1D layer learns a projection that is optimal for the denoising objective, not constrained to sinusoidal bases or perceptual scales.
- **Complementary to STFT.** Stage 1 already handles frequency-domain masking. The learned transform in Stage 2 captures residual distortions and temporal patterns that the Fourier representation misses.
- **Computational efficiency.** Pointwise convolutions (kernel = 1) add negligible computation while providing a fully learnable, end-to-end differentiable transform.

This dual-transform design — one fixed (STFT), one learned (Conv1D) — is what gives DTLN its robustness across diverse noise conditions without requiring domain-specific feature engineering.

---

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample rate | 16 000 Hz | Input audio sample rate |
| Block length | 512 | Frame size for STFT |
| Block shift | 128 | Hop size (75% overlap) |
| LSTM units | 128 | Hidden units per LSTM layer |
| LSTM layers | 2 | Stacked layers per separation kernel |
| Encoder size | 256 | Conv1D output channels |
| Dropout | 0.25 | Applied between LSTM layers |
| Mask activation | Sigmoid | Bounds mask to [0, 1] |

---

## Instant Layer Normalization

A per-frame normalization layer that computes mean and variance over the feature axis:

```
output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

- **gamma** — learnable scale (initialized to 1)
- **beta** — learnable shift (initialized to 0)
- **epsilon** — numerical stability constant (1e-7)

This is applied before each separation kernel to stabilize LSTM input distributions.

---

## Loss Function

The model is trained with a **negative SNR loss**:

```
SNR = mean(s_true^2) / mean((s_true - s_estimate)^2 + 1e-7)
loss = -10 * log10(SNR)
```

Minimizing this loss directly maximizes the signal-to-noise ratio between the estimated and reference clean speech.

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (gradient clipping, max norm = 3.0) |
| Learning rate | 1e-3 (initial) |
| Batch size | 32 |
| Max epochs | 200 |
| Segment length | 15 seconds |

**Callbacks:**

- **ReduceLROnPlateau** — halve LR if val_loss stalls for 3 epochs
- **EarlyStopping** — stop after 10 epochs without improvement
- **ModelCheckpoint** — save best weights only
- **CSVLogger** — record training metrics per epoch

---

## Model Variants

| Variant | Layers | Use case |
|---------|--------|----------|
| Standard | 9 | Batch inference, variable-length input |
| Normalized (norm) | 10 | Adds LayerNorm on STFT magnitude before Stage 1 |
| Stateful | 9 or 10 | Real-time frame-by-frame processing (batch = 1) |
| TF-Lite | Split into 2 models | Mobile / edge deployment with optional quantization |

---

## Usage

```python
from test_package.test_dtln import DTLNModel

model = DTLNModel("model/DTLN_vivos_best.h5", norm_stft=False)
model.build()
model.load_weights()

denoised = model.denoise(noisy_audio, fs=16000)
```

---

## References

- Westhausen & Meyer, *"Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression"*, INTERSPEECH 2020. [arXiv:2005.07551](https://arxiv.org/abs/2005.07551)
- Luo & Mesgarani, *"Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"*, 2019. [arXiv:1809.07454](https://arxiv.org/abs/1809.07454v2)
