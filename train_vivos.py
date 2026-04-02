import os
import argparse
import shutil
import glob
from pathlib import Path
from model import DTLN_model
from tensorflow.keras.callbacks import Callback


class EpochCheckpointCallback(Callback):
    """
    Custom callback to save model after every epoch and track best model.
    Saves:
    - Every epoch: {run_name}_epoch_XXX.h5
    - Latest: {run_name}_latest.h5
    - Best (by val_loss): {run_name}_best.h5
    """
    
    def __init__(self, checkpoint_dir, run_name):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.best_val_loss = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint after each epoch"""
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_epoch_{epoch+1:03d}.h5"
        )
        
        print(f"\nSaving checkpoint: {os.path.basename(epoch_checkpoint_path)}")
        self.model.save_weights(epoch_checkpoint_path)
        
        # Save as latest
        latest_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_latest.h5"
        )
        shutil.copy(epoch_checkpoint_path, latest_checkpoint_path)
        
        # Save best model
        val_loss = logs.get('val_loss')
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.run_name}_best.h5" 
            )
            shutil.copy(epoch_checkpoint_path, best_checkpoint_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        print(f"Checkpoint saved successfully!")


def check_dataset_exists(dataset_path):
    """
    Verify that dataset directories exist and contain files.
    
    Args:
        dataset_path: Base path to dataset directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    required_dirs = [
        'train/noisy',
        'train/clean',
        'val/noisy',
        'val/clean'
    ]
    
    print("=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing: {full_path}")
            all_exist = False
        else:
            # Count .wav files
            wav_files = [f for f in os.listdir(full_path) if f.endswith('.wav')]
            print(f"{dir_path:20s} - {len(wav_files):,} files")
    
    print("=" * 70)
    return all_exist


def evaluate_model(modelTrainer, dataset_path, run_name):
    """
    Evaluate the trained model on test dataset.
    
    Args:
        modelTrainer: Trained DTLN model instance
        dataset_path: Base path to dataset
        run_name: Name of the training run
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 70)
    
    test_noisy = os.path.join(dataset_path, 'test', 'noisy')
    test_clean = os.path.join(dataset_path, 'test', 'clean')
    
    # Check if test set exists
    if not os.path.exists(test_noisy) or not os.path.exists(test_clean):
        print("Test dataset not found. Skipping evaluation.")
        return
    
    # Count test files
    test_files = [f for f in os.listdir(test_noisy) if f.endswith('.wav')]
    print(f"Test files: {len(test_files):,}")
    
    # Load best model
    best_model_path = f"models_{run_name}/{run_name}_best.h5"
    if os.path.exists(best_model_path):
        print(f"\nLoading best model: {best_model_path}")
        modelTrainer.model.load_weights(best_model_path)
    else:
        print(f"\nBest model not found at {best_model_path}")
        print("Using current model weights for evaluation")
    
    # Evaluate on test set (using model.evaluate if available)
    try:
        # Create test data generator (same logic as training)
        from model import audio_generator
        import numpy as np
        
        len_in_samples = int(np.fix(modelTrainer.fs * modelTrainer.len_samples / 
                                    modelTrainer.block_shift) * modelTrainer.block_shift)
        
        generator_test = audio_generator(
            test_noisy,
            test_clean,
            len_in_samples,
            modelTrainer.fs,
            train_flag=False
        )
        
        dataset_test = generator_test.tf_data_set
        dataset_test = dataset_test.batch(modelTrainer.batchsize, drop_remainder=True)
        steps_test = generator_test.total_samples // modelTrainer.batchsize
        
        print(f"Test samples: {generator_test.total_samples:,}")
        print(f"Test steps: {steps_test:,}")
        print("\nRunning evaluation...")
        
        results = modelTrainer.model.evaluate(
            dataset_test,
            steps=steps_test,
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Test Loss (negative SNR): {results:.4f}")
        print(f"Estimated SNR improvement: {-results:.2f} dB")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("You can manually test the model using test_dtln_inference.py")


def main():
    parser = argparse.ArgumentParser(
        description='Train DTLN model on VIVOS + DNS noise dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset paths
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='datasets',
        help='Path to dataset directory (containing train/val/test)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help='Maximum number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--sample-length',
        type=int,
        default=3,  # Changed from 15 to 3 for VIVOS dataset (files are 3-5 seconds)
        help='Length of audio samples in seconds'
    )
    
    # Model configuration
    parser.add_argument(
        '--run-name',
        type=str,
        default='DTLN_vivos',
        help='Name for this training run'
    )
    
    parser.add_argument(
        '--norm-stft',
        action='store_true',
        help='Use STFT normalization (log-magnitude normalization)'
    )
    
    # GPU configuration
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device ID to use (e.g., "0" or "0,1")'
    )
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Activate for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Construct dataset paths
    dataset_path = args.dataset_path
    path_to_train_mix = os.path.join(dataset_path, 'train', 'noisy')
    path_to_train_speech = os.path.join(dataset_path, 'train', 'clean')
    path_to_val_mix = os.path.join(dataset_path, 'val', 'noisy')
    path_to_val_speech = os.path.join(dataset_path, 'val', 'clean')
    
    # Verify dataset exists
    if not check_dataset_exists(dataset_path):
        print("\nDataset validation failed!")
        print("Please ensure the dataset is prepared correctly.")
        print(f"Expected location: {os.path.abspath(dataset_path)}")
        return 1
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Run name:          {args.run_name}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Max epochs:        {args.max_epochs}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Sample length:     {args.sample_length}s")
    print(f"STFT normalization: {args.norm_stft}")
    print(f"GPU device:        {args.gpu}")
    print("=" * 70)
    
    print("\nCreating DTLN model...")
    modelTrainer = DTLN_model()
    
    # Set hyperparameters
    modelTrainer.batchsize = args.batch_size
    modelTrainer.max_epochs = args.max_epochs
    modelTrainer.lr = args.learning_rate
    modelTrainer.len_samples = args.sample_length
    
    print("Building DTLN architecture...")
    modelTrainer.build_DTLN_model(norm_stft=args.norm_stft)
    print("Model built successfully!\n")
    
    print("Compiling model...")
    modelTrainer.compile_model()
    print("Model compiled!\n")
    
    # Train the model with custom checkpoint callback
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Training data:   {path_to_train_mix}")
    print(f"Validation data: {path_to_val_mix}")
    print(f"Models will be saved to: models_{args.run_name}/")
    print("=" * 70)
    print()
    
    try:
        # Import necessary modules for custom training
        from model import audio_generator
        from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
        import numpy as np
        
        savePath = f'./models_{args.run_name}/'
        os.makedirs(savePath, exist_ok=True)
        
        # Calculate sample length
        len_in_samples = int(np.fix(modelTrainer.fs * modelTrainer.len_samples / 
                                    modelTrainer.block_shift) * modelTrainer.block_shift)
        
        print(f"Audio sample length: {len_in_samples} samples ({len_in_samples/modelTrainer.fs:.2f} seconds)")
        print("\nCreating data generators...")
        
        # Create training data generator
        print("   Loading training data...")
        generator_input = audio_generator(
            path_to_train_mix,
            path_to_train_speech,
            len_in_samples,
            modelTrainer.fs,
            train_flag=True
        )
        
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(modelTrainer.batchsize, drop_remainder=True).repeat()
        steps_train = generator_input.total_samples // modelTrainer.batchsize
        
        print(f"   Training samples: {generator_input.total_samples:,}")
        print(f"   Training steps per epoch: {steps_train:,}")
        
        # Create validation data generator
        print("\n   Loading validation data...")
        generator_val = audio_generator(
            path_to_val_mix,
            path_to_val_speech,
            len_in_samples,
            modelTrainer.fs
        )
        
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(modelTrainer.batchsize, drop_remainder=True).repeat()
        steps_val = generator_val.total_samples // modelTrainer.batchsize
        
        print(f"   Validation samples: {generator_val.total_samples:,}")
        print(f"   Validation steps: {steps_val:,}")
        print("\nData generators ready!")
        
        # Setup callbacks
        print("\nSetting up training callbacks...")
        
        # Custom checkpoint callback (saves all epochs + best)
        checkpoint_callback = EpochCheckpointCallback(savePath, args.run_name)
        
        csv_logger = CSVLogger(savePath + f'training_{args.run_name}.log', append=True)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-10,
            cooldown=1,
            verbose=1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1,
            mode='auto'
        )
        
        callbacks = [
            checkpoint_callback,
            csv_logger,
            reduce_lr,
            early_stopping
        ]
        
        print("Callbacks configured!")
        print("\n" + "=" * 70)
        print("Starting model.fit()...")
        print("=" * 70)
        
        modelTrainer.model.fit(
            x=dataset,
            batch_size=None,
            steps_per_epoch=steps_train,
            epochs=modelTrainer.max_epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=steps_val,
            callbacks=callbacks,
            max_queue_size=50,
            workers=4,
            use_multiprocessing=True
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nSaved checkpoints:")
        print(f"   - All epochs: {savePath}{args.run_name}_epoch_XXX.h5")
        print(f"   - Latest: {savePath}{args.run_name}_latest.h5")
        print(f"   - Best: {savePath}{args.run_name}_best.h5")
        print(f"Training log: {savePath}training_{args.run_name}.log")
        
        # Evaluate on test set
        evaluate_model(modelTrainer, dataset_path, args.run_name)
        
        print()
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Latest checkpoint saved to: models_{args.run_name}/")
        print("You can resume training by running this script again.")
        return 130
        
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
