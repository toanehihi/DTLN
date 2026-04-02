# KIẾN TRÚC MÔ HÌNH KHỬ NHIỄU DTLN
*(Dual-signal Transformation LSTM Network)*

---

## 1. Tổng Quan về DTLN

DTLN (Dual-signal Transformation LSTM Network) là một mô hình học sâu được đề xuất bởi Westhausen và Meyer (2020) nhằm giải quyết bài toán tăng cường chất lượng giọng nói (Speech Enhancement) trong môi trường thực tế. Mô hình được thiết kế đặc biệt để có thể chạy hiệu quả trên các thiết bị tài nguyên hạn chế như Raspberry Pi, phù hợp với các ứng dụng thời gian thực.

Điểm đột phá của DTLN nằm ở việc kết hợp hai miền xử lý tín hiệu khác nhau trong một pipeline duy nhất: miền tần số (frequency domain) thông qua biến đổi STFT, và miền đặc trưng học được (learned feature domain) thông qua bộ mã hóa Conv1D. Nhờ kiến trúc hai giai đoạn này, mô hình có khả năng khai thác đồng thời đặc trưng cấu trúc của phổ tần số và các đặc trưng ẩn trong miền đặc trưng.

Trong dự án này, DTLN được huấn luyện trên bộ dữ liệu **VIVOS** (tiếng Việt) kết hợp với tập nhiễu **DNS Challenge**, nhằm tối ưu cho bài toán khử nhiễu giọng nói tiếng Việt trong ứng dụng hỗ trợ bán hàng thông minh nhúng.

### Thông số kỹ thuật tổng quan (theo mã nguồn `model.py`)

| Thông số | Giá trị | Tham chiếu code |
|---|---|---|
| Tần số lấy mẫu (`fs`) | 16,000 Hz | `self.fs = 16000` |
| Kích thước khung STFT (`blockLen`) | 512 mẫu (32 ms) | `self.blockLen = 512` |
| Bước dịch khung (`block_shift`) | 128 mẫu (8 ms), overlap 75% | `self.block_shift = 128` |
| Số bin tần số STFT | 257 (= blockLen//2 + 1) | `rfft` trả về NFFT/2+1 |
| Số đơn vị LSTM mỗi lớp (`numUnits`) | 128 units | `self.numUnits = 128` |
| Số lớp LSTM mỗi stage (`numLayer`) | 2 | `self.numLayer = 2` |
| Dropout giữa các lớp LSTM | 0.25 | `self.dropout = 0.25` |
| Số bộ lọc Encoder (`encoder_size`) | 256 | `self.encoder_size = 256` |
| Kernel size Encoder Conv1D | **1** (point-wise convolution) | `Conv1D(self.encoder_size, 1, ...)` |
| Stride Encoder Conv1D | **1** | `strides=1` |
| Kernel size Decoder Conv1D | **1** (point-wise convolution) | `Conv1D(self.blockLen, 1, ...)` |
| Hàm kích hoạt mask | Sigmoid → [0, 1] | `self.activation = 'sigmoid'` |
| Hàm mất mát | **Negative SNR** (không phải SI-SDR) | `self.cost_function = self.snr_cost` |
| Hàm chuẩn hóa | InstantLayerNormalization (channel-wise) | Class `InstantLayerNormalization` |
| Batch size (VIVOS training) | **32** | `train_vivos.py: default=32` |
| Độ dài mẫu audio (VIVOS) | **3 giây** (phù hợp VIVOS 3–5s) | `train_vivos.py: default=3` |
| Learning rate | 1e-3 (Adam, clipnorm=3.0) | `self.lr = 1e-3` |
| Max epochs | 50 | `self.max_epochs = 50` |
| Mono channel | Có | Chỉ hỗ trợ single channel |

---

## 2. Kiến Trúc Tổng Thể

DTLN bao gồm hai khối xử lý tuần tự (Separation Core), mỗi khối đảm nhận vai trò lọc tín hiệu trong một miền biểu diễn khác nhau. Đầu vào là tín hiệu âm thanh thô chứa giọng nói bị nhiễu, đầu ra là tín hiệu giọng nói đã được phục hồi. Kết quả đầu ra của giai đoạn 1 trở thành đầu vào cho giai đoạn 2.

Toàn bộ mô hình được huấn luyện **end-to-end** bằng hàm mất mát **Negative SNR** (Signal-to-Noise Ratio), cho phép tối ưu hóa đồng thời cả hai giai đoạn mà không cần nhãn trung gian. Optimizer **Adam** với gradient clipping (`clipnorm=3.0`) được sử dụng.

```
Input x(t)  ───  waveform (batch_size, len_in_samples)
    │
    ▼
┌──────────────────────────────────────────────┐
│                 STAGE 1 (STFT Domain)        │
│  STFT (frame→rfft) → [InstantLayerNorm]      │
│  → LSTM×2 (128 units) → Dense(257)+Sigmoid   │
│  → Multiply mask × mag → iFFT               │
└──────────────────┬───────────────────────────┘
                   │ estimated_frames_1 (time-domain frames)
                   ▼
┌──────────────────────────────────────────────┐
│            STAGE 2 (Feature Domain)          │
│  Conv1D Encoder(256, k=1) → InstantLayerNorm │
│  → LSTM×2 (128 units) → Dense(256)+Sigmoid   │
│  → Multiply mask × encoded → Conv1D Dec(512) │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
           Overlap-and-Add
                   │
                   ▼
         Output estimated_sig  ───  waveform sạch
```

---

## 3. Giai Đoạn 1: Xử Lý trong Miền Tần Số (STFT Domain)

### 3.1. Biến Đổi Fourier Thời Gian Ngắn (STFT)

Tín hiệu âm thanh đầu vào được phân khung bằng `tf.signal.frame` với `blockLen=512` và `block_shift=128`, sau đó áp dụng `tf.signal.rfft` để chuyển sang miền tần số:

```python
# model.py - stftLayer()
frames = tf.signal.frame(x, self.blockLen, self.block_shift)
stft_dat = tf.signal.rfft(frames)
mag = tf.abs(stft_dat)        # phổ biên độ, shape: (..., 257)
phase = tf.math.angle(stft_dat)  # pha, giữ nguyên
```

**Lưu ý quan trọng:** Trong cài đặt này, **không sử dụng hàm cửa sổ** (Hanning/Hamming) trước khi tính FFT. Tín hiệu được phân khung trực tiếp từ `tf.signal.frame` mà không áp dụng windowing function. Việc giảm hiệu ứng rò rỉ phổ (spectral leakage) được bù đắp một phần bởi overlap 75% và bộ mã hóa Conv1D ở Stage 2.

Kết quả:
- **Phổ biên độ** `|X(f, t)|`: 257 bin tần số, dùng làm đầu vào cho mạng neural
- **Pha** `∠X(f, t)`: giữ nguyên, tái sử dụng khi tổng hợp lại tín hiệu

### 3.2. Chuẩn Hóa — InstantLayerNormalization (tùy chọn)

Mô hình hỗ trợ chuẩn hóa phổ biên độ qua cờ `norm_stft`:

- **`norm_stft=True`**: Áp dụng `log(mag + 1e-7)` rồi đi qua `InstantLayerNormalization`
- **`norm_stft=False`** (mặc định trong dự án): Phổ biên độ được truyền trực tiếp vào LSTM mà không qua chuẩn hóa

`InstantLayerNormalization` là lớp chuẩn hóa theo channel (channel-wise layer normalization), được đề xuất bởi Luo & Mesgarani (2018). Khác với Instance Normalization thông thường, lớp này chuẩn hóa **trên chiều đặc trưng (feature dimension)** của từng khung thời gian độc lập:

```python
# InstantLayerNormalization.call()
mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
std = tf.math.sqrt(variance + 1e-7)
outputs = (inputs - mean) / std
outputs = outputs * self.gamma + self.beta  # tham số học được
```

### 3.3. Khối LSTM Hai Lớp (Separation Kernel)

Phổ biên độ (đã hoặc chưa chuẩn hóa) được đưa qua **Separation Kernel** — khối xử lý cốt lõi gồm:

```python
# model.py - seperation_kernel()
for idx in range(num_layer):       # num_layer = 2
    x = LSTM(self.numUnits, return_sequences=True)(x)  # 128 units
    if idx < (num_layer - 1):
        x = Dropout(self.dropout)(x)   # dropout = 0.25 giữa lớp 1 và 2
mask = Dense(mask_size)(x)         # mask_size = 257 (STFT bins)
mask = Activation('sigmoid')(mask) # ép giá trị về [0, 1]
```

| Thành phần | Chi tiết |
|---|---|
| LSTM lớp 1 | 128 units, `return_sequences=True` → học đặc trưng cục bộ |
| Dropout | Tỉ lệ 0.25, áp dụng giữa 2 lớp LSTM |
| LSTM lớp 2 | 128 units, `return_sequences=True` → tổng hợp thời gian dài hạn |
| Dense | 257 units (= số bin tần số STFT) |
| Activation | Sigmoid → mask M1 ∈ [0, 1] |

### 3.4. Ước Lượng Mask và Tổng Hợp Tín Hiệu

Mask M1 được nhân element-wise với phổ biên độ gốc, sau đó kết hợp lại với pha gốc qua iFFT:

```python
estimated_mag = Multiply()([mag, mask_1])
# iFFT: tái tạo tín hiệu thời gian từ magnitude + phase
s1_stft = tf.cast(estimated_mag, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
estimated_frames_1 = tf.signal.irfft(s1_stft)  # → time-domain frames (512 samples)
```

Kết quả `estimated_frames_1` là các khung tín hiệu thời gian đã được lọc nhiễu sơ bộ, trở thành đầu vào trực tiếp cho Stage 2.

---

## 4. Giai Đoạn 2: Xử Lý trong Miền Đặc Trưng (Feature Domain)

### 4.1. Bộ Mã Hóa Conv1D (Encoder)

Các khung tín hiệu `estimated_frames_1` (512 chiều) được chiếu sang không gian đặc trưng 256 chiều bằng **Conv1D point-wise** (kernel_size=1):

```python
encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
# encoder_size=256, kernel=1, stride=1, no bias
```

| Tham số | Giá trị (trong dự án) |
|---|---|
| Số bộ lọc | **256** |
| Kernel size | **1** (point-wise, không phải 32) |
| Stride | **1** |
| Bias | **False** |
| Activation | **Không có** (linear projection) |

**Khác biệt quan trọng so với bài báo gốc:** Cài đặt trong dự án sử dụng `kernel_size=1` (point-wise convolution), hoạt động như một phép chiếu tuyến tính (linear projection) từ 512-D sang 256-D trên từng khung thời gian. Đây không phải Conv1D với kernel lớn như mô tả trong bài báo gốc (kernel=32, stride=32). Phép chiếu point-wise cho phép mô hình học một phép biến đổi cơ sở (basis transformation) riêng cho mỗi khung thời gian.

### 4.2. InstantLayerNormalization và LSTM

Biểu diễn mã hóa được chuẩn hóa qua `InstantLayerNormalization` (luôn bật ở Stage 2, không phụ thuộc `norm_stft`), sau đó qua Separation Kernel thứ hai:

```python
encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
# → 2 lớp LSTM(128) + Dense(256) + Sigmoid
```

Việc áp dụng LSTM trong không gian mã hóa giúp mô hình học tái cấu trúc cấu trúc thời gian phức tạp của tín hiệu giọng nói, đặc biệt hữu ích với nhiễu có cấu trúc (âm nhạc nền, tiếng xe cộ).

### 4.3. Ước Lượng Mask và Giải Mã (Decoder)

Đầu ra LSTM → Dense(256) + Sigmoid → mask M2 ∈ [0, 1]:

```python
estimated = Multiply()([encoded_frames, mask_2])
decoded_frames = Conv1D(self.blockLen, 1, padding='causal', use_bias=False)(estimated)
# blockLen=512, kernel=1, causal padding, no bias
```

**Bộ giải mã (Decoder)** sử dụng **Conv1D thông thường** (KHÔNG phải Conv1D Transposed) với `kernel_size=1` và `padding='causal'`, chiếu ngược từ 256-D về 512-D (kích thước khung gốc).

### 4.4. Tái Tạo Tín Hiệu (Overlap-and-Add)

Các khung đã giải mã được ghép lại thành tín hiệu liên tục qua `tf.signal.overlap_and_add`:

```python
estimated_sig = tf.signal.overlap_and_add(decoded_frames, self.block_shift)
# block_shift=128 → ghép các frame chồng lấn thành waveform liên tục
```

---

## 5. Phân Tích Chi Tiết Các Thành Phần Cốt Lõi

### 5.1. Dual-Signal Transformation

| | STFT (Stage 1) | Conv1D Encoder (Stage 2) |
|---|---|---|
| Loại biến đổi | Cố định (`rfft`, fixed basis) | Học được (learnable linear projection) |
| Kernel size | N/A (FFT) | 1 (point-wise) |
| Chiều đầu ra | 257 (frequency bins) | 256 (learned features) |
| Ý nghĩa vật lý | Rõ ràng (từng bin tần số) | Trừu tượng |
| Vai trò | Loại bỏ nhiễu tổng thể ở mức phổ | Tinh chỉnh, phục hồi chi tiết trong đặc trưng ẩn |

### 5.2. Multiplicative Masking

Cả hai stage đều dùng **soft mask** (nhân element-wise) thay vì dự đoán trực tiếp phổ giọng nói:

- **Mask ≈ 1:** Giữ nguyên thành phần đó (giọng nói)
- **Mask ≈ 0:** Loại bỏ hoàn toàn (nhiễu)

Ưu điểm: ổn định hơn spectral mapping, dễ tối ưu hóa, tránh artifacts do dự đoán sai biên độ tuyệt đối.

### 5.3. Hàm Mất Mát — Negative SNR

Dự án sử dụng hàm mất mát **Negative SNR** (Signal-to-Noise Ratio), **không phải SI-SDR** như bài báo gốc:

```python
# model.py - snr_cost()
snr = mean(s_true²) / (mean((s_true - s_estimate)²) + 1e-7)
loss = -10 * log₁₀(snr)
```

Hàm này tính tỉ lệ năng lượng tín hiệu sạch so với năng lượng sai số, chuyển sang đơn vị dB rồi đảo dấu (vì cần minimize loss). Giá trị loss càng thấp (âm hơn) nghĩa là chất lượng khử nhiễu càng tốt.

**So sánh với SI-SDR:** SNR đơn giản hơn và không có bước chiếu tỉ lệ (scale projection) như SI-SDR. Tuy nhiên, SNR vẫn phản ánh tốt chất lượng khử nhiễu tương đối.

---

## 6. Cấu Hình Huấn Luyện trên VIVOS

### Chiến lược huấn luyện (`train_vivos.py`)

| Tham số | Giá trị | Lý do |
|---|---|---|
| Batch size | 32 | Cân bằng bộ nhớ GPU và tính ổn định gradient |
| Sample length | **3 giây** | Phù hợp độ dài file VIVOS (3–5 giây) |
| Max epochs | 50 | Kết hợp với Early Stopping |
| Learning rate | 1e-3 → giảm dần | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=10 | Dừng khi val_loss không cải thiện sau 10 epoch |
| Optimizer | Adam (clipnorm=3.0) | Cắt gradient để tránh exploding gradient |
| Seed cố định | 42 | Đảm bảo tái lập kết quả |

### Cấu trúc dữ liệu

```
datasets/
├── train/
│   ├── noisy/    ← file WAV nhiễu (VIVOS + DNS noise)
│   └── clean/    ← file WAV sạch (VIVOS gốc)
├── val/
│   ├── noisy/
│   └── clean/
└── test/
    ├── noisy/
    └── clean/
```

### Callbacks huấn luyện

- **EpochCheckpointCallback** (custom): Lưu checkpoint mọi epoch + best model theo val_loss
- **ReduceLROnPlateau**: Giảm LR 50% khi val_loss không cải thiện sau 3 epoch
- **EarlyStopping**: Dừng sau 10 epoch không cải thiện
- **CSVLogger**: Ghi log huấn luyện

---

## 7. So Sánh Kiến Trúc với Các Phương Pháp Liên Quan

| Đặc điểm | DTLN (dự án) | RNNoise | DeepSpeech Filter | Conv-TasNet |
|---|---|---|---|---|
| Miền xử lý | Kép (STFT + Feature) | Tần số | Tần số | Thời gian |
| Loại RNN | LSTM 2 lớp × 2 stage | GRU | LSTM | Không (Conv) |
| Số tham số | ~1.8M | ~0.06M | ~5M | ~5.1M |
| Real-time | Có (32ms) | Có (<10ms) | Hạn chế | Có |
| Thiết bị nhúng | Có (RPi 4) | Có (ARM) | Không | Hạn chế |
| Dataset tiếng Việt | ✅ VIVOS + DNS | ❌ | ❌ | ❌ |

---

## 8. Pipeline Triển Khai Thực Tế

Mô hình hỗ trợ xuất sang **TF Lite** (2 sub-model) hoặc **SavedModel** cho suy luận:

```
                    ┌─ TF Lite Model 1 ─┐    ┌─ TF Lite Model 2 ─┐
                    │  mag → LSTM×2      │    │  frame → Conv1D    │
Microphone ──►      │  → mask_1          │    │  → LSTM×2 → mask_2│
  Framing(512)      │  + states in/out   │    │  → decode          │
      │             └────────┬───────────┘    │  + states in/out   │
      ▼                      │                └────────┬───────────┘
  rfft (mag, phase)          ▼                         │
      │              mask × mag + phase                ▼
      └──────────►    → irfft → frame ──────────►  clean frame
                                                       │
                                              Overlap-and-Add
                                                       │
                                                       ▼
                                               Clean Audio Output
```

Trong chế độ stateful (`build_DTLN_model_stateful`):
- Mô hình xử lý **từng khung 512 mẫu** (32ms)
- Trạng thái ẩn LSTM (h, c) được **lưu giữ giữa các lần suy luận**
- Hỗ trợ chuyển đổi sang TF Lite với 2 sub-model riêng biệt (Core 1 và Core 2)
- Tùy chọn **dynamic range quantization** để giảm kích thước model

---

## 9. Đánh Giá Ưu Điểm và Hạn Chế

### Ưu Điểm

- **Kiến trúc nhẹ (~1.8M params):** Cạnh tranh với mô hình lớn hơn, phù hợp thiết bị nhúng
- **Real-time (32ms latency):** Đáp ứng yêu cầu giao tiếp tương tác
- **Dual-domain processing:** Khử nhiễu toàn diện hơn mô hình đơn miền
- **Huấn luyện trên tiếng Việt:** Tối ưu cho VIVOS dataset, phù hợp ứng dụng thực tế Việt Nam
- **TF Lite / SavedModel export:** Tương thích triển khai trên nhiều nền tảng phần cứng
- **Stateful inference:** Xử lý streaming real-time với trạng thái LSTM liên tục

### Hạn Chế

- **Mono channel only:** Chưa tận dụng thông tin không gian từ beamforming
- **Nhạy với nhiễu phi tĩnh:** Hiệu năng giảm với nhiễu biến đổi nhanh hoặc có cấu trúc âm nhạc
- **Cần dữ liệu song song:** Yêu cầu cặp (noisy, clean), khó thu thập thực tế với tiếng Việt
- **Pha không được tối ưu hóa:** Stage 1 tái sử dụng pha gốc, có thể gây artifact khi nhiễu nặng
- **Không sử dụng windowing:** STFT không áp dụng hàm cửa sổ, có thể gây spectral leakage

---

## Kết Luận

DTLN là kiến trúc khử nhiễu giọng nói hiệu quả, kết hợp xử lý miền tần số (STFT) và miền đặc trưng học được (Conv1D point-wise projection). Cơ chế hai giai đoạn với LSTM và Multiplicative Masking cho phép mô hình phân tách giọng nói khỏi nhiễu từ hai góc độ biểu diễn khác nhau.

Với kích thước nhỏ gọn (~1.8M tham số), khả năng xử lý thời gian thực (32ms latency), hỗ trợ xuất TF Lite, và đã được huấn luyện trên bộ dữ liệu VIVOS tiếng Việt, DTLN là lựa chọn phù hợp cho hệ thống hỗ trợ bán hàng thông minh nhúng, nơi yêu cầu cân bằng giữa hiệu năng khử nhiễu và tài nguyên tính toán hạn chế.

---

## Tài Liệu Tham Khảo

[1] Westhausen, N. L., & Meyer, B. T. (2020). Dual-signal transformation LSTM network for real-time noise suppression. *Proc. Interspeech 2020*, 2477–2481.

[2] Valentini-Botinhao, C., et al. (2016). Investigating RNN-based speech enhancement methods for noise-robust text-to-speech. *SSW 2016*.

[3] Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation. *IEEE/ACM TASLP*, 27(8), 1256–1266.

[4] Luo, Y., & Mesgarani, N. (2018). TasNet: time-domain audio separation network. *arXiv:1809.07454v2* — nguồn gốc InstantLayerNormalization.

[5] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735–1780.