"""
Training Script for RIGID Gait Recognition Model

This script handles the complete training pipeline for the RIGID model,
including data loading, model compilation, training, and evaluation.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.rigid_model import create_rigid_model, RIGIDModel
from data.preprocessing import load_processed_data, GaitDataProcessor
from utils.metrics import calculate_metrics, plot_training_history
from utils.visualization import plot_confusion_matrix


class RIGIDTrainer:
    """Training manager for RIGID gait recognition model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
        
        # Create output directories
        self.create_output_dirs()
        
        # Setup GPU if available
        self.setup_gpu()
    
    def create_output_dirs(self):
        """Create necessary output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.model_dir = self.output_dir / 'models'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.model_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_gpu(self):
        """Setup GPU configuration."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU available, using CPU")
    
    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """
        Load and preprocess training data.
        
        Returns:
            Tuple of (train_dataset, val_dataset, num_classes)
        """
        data_config = self.config['data']
        
        if data_config.get('use_preprocessed', True):
            # Load preprocessed data
            print("Loading preprocessed data...")
            coord_maps, dist_matrices, gray_frames, labels = load_processed_data(
                data_config['preprocessed_path']
            )
        else:
            # Process raw videos
            print("Processing raw video data...")
            processor = GaitDataProcessor(
                frame_size=data_config['frame_size']
            )
            
            features, labels = processor.process_dataset(
                dataset_path=data_config['dataset_path'],
                max_frames_per_video=data_config['max_frames_per_video'],
                max_videos_per_class=data_config.get('max_videos_per_class')
            )
            
            # Convert to numpy arrays
            coord_maps = np.array([f['coordinate_maps'] for f in features])
            dist_matrices = np.array([f['distance_matrices'] for f in features])
            gray_frames = np.array([f['grayscale_frames'] for f in features])
            labels = np.array(labels)
        
        # Calculate number of classes
        num_classes = len(np.unique(labels))
        print(f"Number of classes: {num_classes}")
        print(f"Total samples: {len(labels)}")
        
        # Split data
        split_ratio = data_config.get('validation_split', 0.2)
        split_idx = int(len(labels) * (1 - split_ratio))
        
        # Shuffle data
        indices = np.random.permutation(len(labels))
        coord_maps = coord_maps[indices]
        dist_matrices = dist_matrices[indices]
        gray_frames = gray_frames[indices]
        labels = labels[indices]
        
        # Split train/validation
        train_data = {
            'coordinate_maps': coord_maps[:split_idx],
            'distance_matrices': dist_matrices[:split_idx],
            'grayscale_frames': gray_frames[:split_idx]
        }
        train_labels = labels[:split_idx]
        
        val_data = {
            'coordinate_maps': coord_maps[split_idx:],
            'distance_matrices': dist_matrices[split_idx:],
            'grayscale_frames': gray_frames[split_idx:]
        }
        val_labels = labels[split_idx:]
        
        print(f"Train samples: {len(train_labels)}")
        print(f"Validation samples: {len(val_labels)}")
        
        # Create TensorFlow datasets
        batch_size = self.config['training']['batch_size']
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, num_classes
    
    def create_model(self, num_classes: int) -> tf.keras.Model:
        """Create and compile RIGID model."""
        model_config = self.config['model']
        
        self.model = create_rigid_model(
            input_shape=(
                self.config['data']['frame_size'],
                self.config['data']['frame_size'], 
                3
            ),
            num_classes=num_classes,
            model_size=model_config.get('model_size', 'small'),
            frame_size=self.config['data']['frame_size']
        )
        
        # Compile model
        training_config = self.config['training']
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_config['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_dir / 'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training'].get('early_stopping_patience', 20),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.logs_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self):
        """Main training function."""
        print("Starting RIGID model training...")
        
        # Load data
        train_dataset, val_dataset, num_classes = self.load_data()
        
        # Create model
        model = self.create_model(num_classes)
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Training configuration
        training_config = self.config['training']
        
        # Start training
        print(f"\nStarting training for {training_config['epochs']} epochs...")
        
        self.history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.model_dir / 'final_model.h5'
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Evaluate model
        self.evaluate_model(model, val_dataset)
        
        # Plot training history
        self.plot_training_history()
        
        return model
    
    def evaluate_model(self, model: tf.keras.Model, val_dataset: tf.data.Dataset):
        """Evaluate model performance."""
        print("\nEvaluating model...")
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for batch_data, batch_labels in val_dataset:
            predictions = model.predict(batch_data, verbose=0)
            y_true.extend(batch_labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1-Score: {metrics['f1_score']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}")
        print(f"Validation Recall: {metrics['recall']:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, 
            save_path=self.plots_dir / 'confusion_matrix.png'
        )
        
        # Save metrics
        metrics_path = self.plots_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    
    def plot_training_history(self):
        """Plot and save training history."""
        if self.history is None:
            print("No training history available")
            return
        
        plot_training_history(
            self.history.history,
            save_path=self.plots_dir / 'training_history.png'
        )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RIGID gait recognition model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset_path:
        config['data']['dataset_path'] = args.dataset_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['output_dir'] = Path(config['output_dir']) / f"rigid_training_{timestamp}"
    
    print(f"Configuration loaded from {args.config}")
    print(f"Output directory: {config['output_dir']}")
    
    # Create trainer and start training
    trainer = RIGIDTrainer(config)
    model = trainer.train()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
