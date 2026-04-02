# Kiến trúc mô hình DTLN (Dual-signal Transformation LSTM Network)

Dựa trên mã nguồn `model.py`, lớp `DTLN_model` xây dựng kiến trúc mạng xử lý giảm nhiễu (noise reduction) hiệu quả theo thời gian thực. DTLN xử lý âm thanh qua hai phần chính (Core 1 và Core 2), kết hợp cả miền tần số (STFT) và miền đặc trưng (dùng 1D Conv).

Dưới đây là giải thích chi tiết từng layer và luồng xử lý (pipeline) của mô hình.

---

## 1. Đầu vào (Input Layer)
**Code:** `time_dat = Input(batch_shape=(None, None))`
*(Hoặc `(1, self.blockLen)` đối với dạng stateful real-time).*

* **Giải thích:** Lớp đầu vào nhận tín hiệu dòng thời gian có độ phân giải cao (ví dụ: tần số lấy mẫu là 16kHz).
* **Ví dụ trong Pipeline:** Đầu vào là một đoạn âm thanh liên tục dạng sóng (waveform) 1 chiều chứa cả giọng nói người và tiếng ồn nền, có kích thước mảng là `(batch_size, sequence_length)`. Với phiên bản triển khai thời gian thực (stateful), mô hình nhận lần lượt từng khung âm thanh có độ dài `blockLen=512` sample (tương đương với cắt từng đoạn 32ms âm thanh tại mỗi 16kHz).

---

## 2. STFT Transformation (Chuyển đổi sang miền tần số)
**Code:** `mag, angle = Lambda(self.stftLayer)(time_dat)`

* **Giải thích:** Áp dụng phép biến đổi Fourier thời gian ngắn (STFT) để chia tín hiệu âm thanh thành các khung (frames) với độ trượt (overlap). Ở đây `blockLen=512` và dịch khung `block_shift=128`. Hàm biến đổi trả về hai thành phần quan trọng: Biên độ (`mag`) và Pha (`angle`).
* **Ví dụ trong Pipeline:** Khung âm thanh 512 mẫu (sample) đi qua layer này sinh ra **phổ biên độ `mag`** có kích thước 257 bins (từ nửa mảng 512 do tính chất đối xứng rFFT) biểu thị cho năng lượng ở các dải tần số. Phần tỷ lệ góc **pha `angle`** sẽ được giữ nguyên mà không đi qua học sâu, vì việc tác động vào pha sẽ làm âm thanh bị méo tiếng rất nặng.

---

## 3. Lõi Phân Tách 1 (Core 1 - STFT Domain)
Phần này áp dụng khối Separation Kernel bằng mạng học sâu LSTM để tính toán mask trên phổ biên độ nhằm loại bỏ nhiễu trên bề mặt tần số.

### 3.1. Chuẩn hóa (Layer Normalization)
**Code:** `mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))`

* **Giải thích:** Nếu được kích hoạt `norm_stft=True`, biên độ `mag` được đưa vào hàm logarit để nén dải tín hiệu. Công cụ **Instant Layer Normalization** (chuẩn hóa lớp theo thời gian thực) sẽ tính trung bình và độ lệch chuẩn để chuẩn hóa giá trị về tỷ lệ chuẩn trên từng khung độc lập.
* **Ví dụ:** Bước này giúp cường độ âm thanh (âm lượng to/nhỏ khác nhau) được đưa về cùng một tầm phân phối ổn định. Khi một tiếng sét nổi lên đột ngột (nhiễu xung), khung chứa tiếng sét sẽ bị kéo giảm cường độ và bị dập tắt dễ hơn.

### 3.2. Mạng LSTM dự đoán Mặt nạ (Separation Kernel 1)
**Code:** `mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen//2+1), mag_norm)`

* **Giải thích:** Hàm `seperation_kernel` bao gồm mạng chuỗi thời gian phân tách gồm:
  * Hai tầng RNN `LSTM(128 units)`.
  * Có xen kẽ tầng `Dropout(0.25)` để tránh học vẹt (overfitting).
  * Cuối cùng là tầng `Dense(257)` nối với hàm kích hoạt `Sigmoid` (ép giá trị về dải [0, 1]).
* **Ví dụ:** Tại mỗi khung thời gian trôi qua, layer LSTM này theo dõi chuỗi `mag_norm` và đưa ra dự đoán `mask_1` có dải giá trị từ 0 đến 1 ở 257 tần số. Nếu mô hình nhận biết ở dải 1200Hz chỉ có rung động của động cơ xe, mask ở vùng 1200Hz sẽ gần giá trị 0. Nếu dải 3000Hz nhận ra giọng nói, mask sẽ bằng 1.

### 3.3. Áp dụng Mask và IFFT (Nghịch đảo STFT)
**Code:** 
```python
estimated_mag = Multiply()([mag, mask_1])
estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])
```

* **Giải thích:** Nhân thẳng `mask_1` vào phổ biên độ `mag` gốc (những vùng bị nhiễu do bị nhân với 0 nên sẽ bị loại bỏ năng lượng). Sau đó dùng rIFFT kết hợp với pha gốc `angle` để chuyển đổi mảng tần số ngược về thành tín hiệu sóng thời gian (time-domain).
* **Ví dụ:** Sinh ra kết quả là sóng âm thanh dạng mảng 512 mẫu số đại diện cho âm thanh sạch tạm thời `estimated_frames_1`. Âm thanh lúc này đã giảm ồn đáng kể, nhưng có thể gặp hiện tượng dội âm nhân tạo hoặc sai lệch dạng sóng do bỏ qua tính toán của Pha.

---

## 4. Lõi Phân Tách 2 (Core 2 - Feature Domain)
Lõi thứ 2 (Core 2) sẽ sửa lỗi và làm mịn đoạn âm thanh trung gian đã lọc bằng Core 1 ở trên. Việc xử lý trên không gian trừu tượng 1D-Conv giải quyết mạnh mẽ nhược điểm khuyết mảng/pha của phần STFT đầu tiên.

### 4.1. 1D-Convolution (Không gian hóa vùng đặc trưng)
**Code:** `encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)`

* **Giải thích:** Chuyển tín hiệu sóng thời gian `estimated_frames_1` (kích thước 512 chiều) sang một không gian đặc trưng biểu diễn mới `encoded_frames` thông qua tầng tích chập `Conv1D (kernel_size=1)` tạo thành mảng có kích kích thước 256.
* **Ví dụ:** Tại bước này, hệ thống sẽ tự học một hệ số bộ lọc số thích ứng dành riêng để phát hiện tàn dư của tiếng ồn hay nhược điểm dạng sóng còn sót lại.

### 4.2. Mạng LSTM dự đoán Mặt nạ (Separation Kernel 2)
**Code:** 
```python
encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
```

* **Giải thích:** Giống với cơ chế lọc ở Core 1. Lại áp dụng chức năng đối xứng: chuẩn hóa `InstantLayerNormalization` và gửi dữ liệu đi vào cấu trúc 2 tầng `LSTM`. Ở đoạn cối là một tầng `Dense(256)` kèm kích hoạt `Sigmoid`.
* **Ví dụ:** Mô hình sẽ tập trung quan sát vào 256 đặc trưng ẩn ở trên mạng Conv1D tạo ra, và tự trả về một mảng `mask_2` để biết kênh đặc trưng nào đang chứa nhiễu.

### 4.3. Mask 2 & Kết hợp 1D-Conv (Decoder)
**Code:**
```python
estimated = Multiply()([encoded_frames, mask_2]) 
decoded_frames = Conv1D(self.blockLen, 1, padding='causal', use_bias=False)(estimated)
```

* **Giải thích:** Tiến hành lọc nhiễu ở Core 2 bằng việc dập `mask_2` vào `encoded_frames`. Cuối cùng, tầng `Conv1D` giải mã (decoder) từ 256 vector đặc trưng khôi phục lại mảng độ dài chuỗi nguyên bản ban đầu là `blockLen=512`.
* **Ví dụ:** Output xuất ra lúc này `decoded_frames` có 512 mẫu số âm thanh sạch (cho 32ms) nhưng cần được dán nối để thành chuỗi âm thanh cho 1 file audio.

---

## 5. Tái tạo Âm Thanh Toàn Diện (Overlap-And-Add)
**Code:** `estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)`

* **Giải thích:** Quá trình STFT lúc phân tách ban đầu đã tiến hành lấy mẫu chồng chéo (overlap). Bước này sử dụng lại phương thức Overlap-and-Add cộng dồn các mảng trượt chồng lên nhau, khử đi các cạnh biên để ghép các frame rời rạc thành chuỗi dữ liệu sóng mượt mà xuyên suốt liên tục.
* **Ví dụ trong Pipeline:** Đầu ra cuối cùng `estimated_sig` là tín hiệu âm thanh dạng sóng (Waveform array) có kích thước độ dài tương đồng với mảng đầu vào ở `Input Layer`. Nhưng giờ đây, tín hiệu tiếng loa rè, tiếng xe cơ giới, hay âm thanh bát đĩa ở quán cafe đã bị triệt tiêu rõ rệt, để lấy ra giọng nói trong sạch. Đầu ra này sẽ được đẩy xuống bộ nhớ hoặc lưu thành file âm thanh sạch `.wav` cuối cùng ở pipeline code của file `evaluate.py`.
