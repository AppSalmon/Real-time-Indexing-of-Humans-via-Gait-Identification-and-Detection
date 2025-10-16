"""
Visualization Utilities for RIGID Gait Recognition

This module provides visualization tools for gait recognition models and data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import List, Dict, Tuple, Optional
import mediapipe as mp


def visualize_pose_landmarks(
    frame: np.ndarray, 
    landmarks: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize pose landmarks on frame.
    
    Args:
        frame: Input frame
        landmarks: Pose landmarks (33, 3)
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Plot landmarks
    for i, landmark in enumerate(landmarks):
        x, y, visibility = landmark
        if visibility > 0.5:  # Only plot visible landmarks
            ax.scatter(x * frame.shape[1], y * frame.shape[0], 
                      c='red', s=20, alpha=0.8)
            ax.annotate(str(i), (x * frame.shape[1], y * frame.shape[0]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, color='white')
    
    ax.set_title('Pose Landmarks Visualization')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pose visualization saved to {save_path}")
    
    return fig


def visualize_tri_channel_features(
    coordinate_map: np.ndarray,
    distance_matrix: np.ndarray, 
    grayscale_frame: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualize tri-channel features.
    
    Args:
        coordinate_map: Coordinate map (H, W, 1)
        distance_matrix: Distance matrix (H, W, 1)
        grayscale_frame: Grayscale frame (H, W, 1)
        save_path: Path to save the visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Coordinate map
    axes[0].imshow(coordinate_map[:, :, 0], cmap='hot')
    axes[0].set_title('Coordinate Map (P)')
    axes[0].axis('off')
    
    # Distance matrix
    axes[1].imshow(distance_matrix[:, :, 0], cmap='viridis')
    axes[1].set_title('Distance Matrix (J)')
    axes[1].axis('off')
    
    # Grayscale frame
    axes[2].imshow(grayscale_frame[:, :, 0], cmap='gray')
    axes[2].set_title('Grayscale Frame (D)')
    axes[2].axis('off')
    
    plt.suptitle('RIGID Tri-Channel Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tri-channel features visualization saved to {save_path}")
    
    return fig


def visualize_model_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize model predictions.
    
    Args:
        images: Input images (batch_size, H, W, C)
        predictions: Model predictions (batch_size, num_classes)
        true_labels: True labels (batch_size,)
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # Display image (assuming grayscale channel)
        if images[i].shape[-1] == 1:
            ax.imshow(images[i][:, :, 0], cmap='gray')
        else:
            ax.imshow(images[i])
        
        # Set title with prediction info
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predicted_classes[i]]
        confidence = confidence_scores[i]
        
        color = 'green' if true_labels[i] == predicted_classes[i] else 'red'
        
        ax.set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model predictions visualization saved to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_names)), sorted_scores, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(sorted_names))))
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Score')
    ax.set_title('Feature Importance Analysis')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, sorted_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def visualize_gait_sequence(
    frames: List[np.ndarray],
    poses: List[np.ndarray],
    title: str = "Gait Sequence",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 4)
) -> plt.Figure:
    """
    Visualize gait sequence with pose overlays.
    
    Args:
        frames: List of frames
        poses: List of pose landmarks
        title: Plot title
        save_path: Path to save the visualization
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_frames = min(len(frames), 10)  # Show max 10 frames
    
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        ax = axes[i]
        
        # Display frame
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        
        # Plot pose landmarks
        if poses[i] is not None:
            for j, landmark in enumerate(poses[i]):
                x, y, visibility = landmark
                if visibility > 0.5:
                    ax.scatter(x * frames[i].shape[1], y * frames[i].shape[0], 
                              c='red', s=10, alpha=0.8)
        
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gait sequence visualization saved to {save_path}")
    
    return fig


def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot model comparison.
    
    Args:
        model_names: List of model names
        metrics: Dictionary of metrics
        metric_name: Name of metric to plot
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = metrics[metric_name]
    
    bars = ax.bar(model_names, values, 
                 color=plt.cm.Set3(np.linspace(0, 1, len(model_names))))
    
    ax.set_xlabel('Models')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Model Comparison - {metric_name}')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully!")
    
    # Example usage
    test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_landmarks = np.random.rand(33, 3)
    
    fig = visualize_pose_landmarks(test_frame, test_landmarks)
    plt.show()
