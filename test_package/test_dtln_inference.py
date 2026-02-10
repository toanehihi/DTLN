#!/usr/bin/env python3
"""
DTLN Model Inference Test Script

This script demonstrates how to use the trained DTLN model to denoise audio.
It loads the model weights and processes a noisy audio file.
"""

import os
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Lambda, Multiply, Activation, Conv1D, Layer


class InstantLayerNormalization(Layer):
    '''Instant layer normalization layer'''
    def __init__(self, **kwargs):
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(shape=shape, initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', trainable=True, name='beta')

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta
        return outputs


class DTLNInference:
    """DTLN Model for Inference"""
    
    def __init__(self, weights_path, load_full_model=False, norm_stft=False):
        """
        Initialize DTLN model for inference
        
        Args:
            weights_path: Path to trained model (.h5 or .weights.h5)
            load_full_model: If True, load complete model from .h5 file.
                           If False, build architecture and load weights.
            norm_stft: If True, use normalized STFT variant (for pretrained models with 'norm' in name)
        """
        self.fs = 16000
        self.blockLen = 512
        self.block_shift = 128
        self.numUnits = 128
        self.numLayer = 2
        self.encoder_size = 256
        self.activation = 'sigmoid'
        self.load_full_model = load_full_model
        self.norm_stft = norm_stft
        self.dropout = 0.25
        
        self.weights_path = weights_path
        self.model = None
        
        print(f"🎯 DTLN Inference Model")
        print(f"   Sample rate: {self.fs} Hz")
        print(f"   Block length: {self.blockLen}")
        print(f"   Block shift: {self.block_shift}")
        
    def stftLayer(self, x):
        '''STFT layer'''
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        stft_dat = tf.signal.rfft(frames)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]
    
    def ifftLayer(self, x):
        '''Inverse FFT layer'''
        s1_stft = (tf.cast(x[0], tf.complex64) * tf.exp((1j * tf.cast(x[1], tf.complex64))))
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        '''Overlap and add layer'''
        return tf.signal.overlap_and_add(x, self.block_shift)

    def seperation_kernel(self, num_layer, mask_size, x):
        '''Separation kernel with LSTM layers'''
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True)(x)
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask

    def build_model(self):
        '''Build DTLN model architecture'''
        print("\n🏗️  Building DTLN model...")
        if self.norm_stft:
            print("   Using NORMALIZED STFT variant (10 layers)")
        else:
            print("   Using standard variant (9 layers)")
        
        time_dat = Input(batch_shape=(None, None))
        mag, angle = Lambda(self.stftLayer)(time_dat)
        
        # First separation kernel
        if self.norm_stft:
            # Normalized variant - add normalization layer
            mag_norm = InstantLayerNormalization()(mag)
        else:
            mag_norm = mag
        
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen // 2 + 1), mag_norm)
        estimated_mag = Multiply()([mag, mask_1])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])
        
        # Encoder
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        
        # Second separation kernel
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
        estimated = Multiply()([encoded_frames, mask_2])
        
        # Decoder
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal', use_bias=False)(estimated)
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)
        
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        print("✅ Model built successfully!")
        
    def load_weights(self):
        '''Load trained weights or full model'''
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        print(f"\n📥 Loading from: {self.weights_path}")
        
        if self.load_full_model:
            # Try to load complete model
            print("   Attempting to load FULL MODEL...")
            try:
                self.model = keras.models.load_model(self.weights_path, compile=False)
                print("✅ Full model loaded successfully!")
            except ValueError as e:
                if "No model config found" in str(e):
                    print("\n❌ Error: This .h5 file contains only weights, not a full model.")
                    print("   The file has a different architecture (10 layers vs 9 layers expected).")
                    print("\n💡 Solutions:")
                    print("   1. Train your own model using the training notebook")
                    print("   2. Use the original DTLN repository code for this pretrained model")
                    print("   3. Download a compatible model from the DTLN repository")
                    raise SystemExit(1)
                else:
                    raise
        else:
            # Load only weights (model architecture must match)
            print("   Loading WEIGHTS ONLY...")
            try:
                self.model.load_weights(self.weights_path)
                print("✅ Weights loaded successfully!")
            except ValueError as e:
                if "Layer count mismatch" in str(e):
                    print(f"\n❌ Error: {e}")
                    print("\n💡 This weights file was trained with a different model architecture.")
                    print("   Please use a model trained with this codebase.")
                    raise SystemExit(1)
                else:
                    raise
        
    def process_audio(self, input_audio_path, output_audio_path):
        '''
        Process noisy audio file and save denoised output
        
        Args:
            input_audio_path: Path to noisy audio file
            output_audio_path: Path to save denoised audio
        '''
        print(f"\n🎵 Processing audio...")
        print(f"   Input:  {input_audio_path}")
        print(f"   Output: {output_audio_path}")
        
        # Read audio
        audio, fs = sf.read(input_audio_path)
        
        if fs != self.fs:
            print(f"   ⚠️  Warning: Sample rate mismatch ({fs} Hz). Resampling to {self.fs} Hz...")
            # Simple resampling (for production, use librosa.resample)
            import scipy.signal
            audio = scipy.signal.resample(audio, int(len(audio) * self.fs / fs))
            fs = self.fs
        
        # Ensure mono
        if audio.ndim > 1:
            print(f"   ⚠️  Converting stereo to mono...")
            audio = np.mean(audio, axis=1)
        
        original_length = len(audio)
        print(f"   Duration: {original_length / fs:.2f} seconds ({original_length} samples)")
        
        # Pad to multiple of block_shift
        pad_length = int(np.ceil(len(audio) / self.block_shift) * self.block_shift)
        if pad_length > len(audio):
            audio = np.pad(audio, (0, pad_length - len(audio)), mode='constant')
            print(f"   Padded to {pad_length} samples")
        
        # Normalize input
        audio = audio.astype('float32')
        
        # Add batch dimension and process
        audio_batch = np.expand_dims(audio, axis=0)
        
        print(f"   Processing with DTLN model...")
        denoised = self.model.predict(audio_batch, verbose=0)
        
        # Remove batch dimension
        denoised = denoised[0]
        
        # Trim to original length
        denoised = denoised[:original_length]
        
        # Save output
        sf.write(output_audio_path, denoised, self.fs)
        print(f"   ✅ Denoised audio saved!")
        
        # Calculate some metrics
        print(f"\n📊 Statistics:")
        print(f"   Input RMS:  {np.sqrt(np.mean(audio[:original_length]**2)):.6f}")
        print(f"   Output RMS: {np.sqrt(np.mean(denoised**2)):.6f}")
        print(f"   Max value:  {np.max(np.abs(denoised)):.6f}")
        

def main():
    """Main inference function"""
    
    print("=" * 70)
    print("  DTLN MODEL INFERENCE TEST")
    print("=" * 70)
    
    # Get script directory (so paths work from anywhere)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths relative to script location
    weights_path = os.path.join(script_dir, "model/DTLN_vivos_best.weights.h5")
    input_audio = os.path.join(script_dir, "test_audio/noisy_sample.wav")
    output_audio = os.path.join(script_dir, "test_audio/denoised_output.wav")
    
    # Check files exist
    if not os.path.exists(weights_path):
        print(f"\n❌ Error: Model weights not found at {weights_path}")
        print(f"   Please ensure the trained model is in the 'model/' directory")
        print(f"   Expected file: DTLN_vivos_best.h5 (NOT .weights.h5)")
        sys.exit(1)
    
    if not os.path.exists(input_audio):
        print(f"\n❌ Error: Test audio not found at {input_audio}")
        print(f"   Please ensure test audio is in the 'test_audio/' directory")
        sys.exit(1)
    
    # Auto-detect model variant from filename
    # Pretrained models like DTLN_norm_500h.h5 are weights-only, need to build architecture
    use_norm = 'norm' in os.path.basename(weights_path).lower()
    
    print(f"\n💡 Model: {os.path.basename(weights_path)}")
    if use_norm:
        print(f"   � Detected NORMALIZED variant")
        print(f"   Will build 10-layer architecture with InstantLayerNormalization")
    else:
        print(f"   Standard variant")
        print(f"   Will build 9-layer architecture")
    
    # Create inference model and build architecture
    dtln = DTLNInference(weights_path, load_full_model=False, norm_stft=use_norm)
    dtln.build_model()
    
    # Load weights
    dtln.load_weights()
    
    # Process audio
    dtln.process_audio(input_audio, output_audio)
    
    print("\n" + "=" * 70)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output saved to: {output_audio}")
    print(f"🎧 You can now listen to the denoised audio!")
    print()


if __name__ == "__main__":
    main()
