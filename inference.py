"""
Inference Script for RIGID Gait Recognition Model

This script handles real-time inference using trained RIGID model.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.rigid_model import load_rigid_model
from data.preprocessing import PoseExtractor, TriChannelGenerator


class RIGIDInference:
    """RIGID model inference handler."""
    
    def __init__(self, model_path: str, frame_size: int = 66):
        """
        Initialize RIGID inference.
        
        Args:
            model_path: Path to trained model
            frame_size: Input frame size
        """
        self.frame_size = frame_size
        self.model = load_rigid_model(model_path)
        self.pose_extractor = PoseExtractor()
        self.tri_channel_generator = TriChannelGenerator(frame_size)
        
        # Class names (update based on your dataset)
        self.class_names = [f'Person_{i+1}' for i in range(64)]  # DMGait dataset has 64 participants
        
        print(f"RIGID model loaded from {model_path}")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed tri-channel features or None if pose not detected
        """
        # Extract pose landmarks
        pose = self.pose_extractor.extract_pose_from_frame(frame)
        
        if pose is None:
            return None
        
        # Generate tri-channel features
        features = self.tri_channel_generator.generate_tri_channel_features(pose, frame)
        
        return features
    
    def predict_single_frame(self, frame: np.ndarray) -> Tuple[int, float, str]:
        """
        Predict gait class for single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (predicted_class, confidence, class_name)
        """
        # Preprocess frame
        features = self.preprocess_frame(frame)
        
        if features is None:
            return -1, 0.0, "No pose detected"
        
        # Prepare input
        input_data = {
            'coordinate_maps': np.expand_dims(features['coordinate_maps'], axis=0),
            'distance_matrices': np.expand_dims(features['distance_matrices'], axis=0),
            'grayscale_frames': np.expand_dims(features['grayscale_frames'], axis=0)
        }
        
        # Predict
        predictions = self.model.predict(input_data, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class_{predicted_class}"
        
        return predicted_class, confidence, class_name
    
    def predict_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Predict gait class for entire video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            
        Returns:
            Dictionary with prediction results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        confidences = []
        frame_times = []
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Predict
            pred_class, confidence, class_name = self.predict_single_frame(frame)
            
            inference_time = time.time() - start_time
            
            predictions.append(pred_class)
            confidences.append(confidence)
            frame_times.append(inference_time)
            
            # Draw prediction on frame
            if pred_class != -1:
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No pose detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame if output specified
            if writer:
                writer.write(frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        # Calculate statistics
        valid_predictions = [p for p in predictions if p != -1]
        avg_confidence = np.mean([c for c in confidences if c > 0])
        avg_inference_time = np.mean(frame_times)
        
        # Most common prediction
        if valid_predictions:
            from collections import Counter
            most_common = Counter(valid_predictions).most_common(1)[0]
            final_prediction = most_common[0]
            prediction_confidence = most_common[1] / len(valid_predictions)
        else:
            final_prediction = -1
            prediction_confidence = 0.0
        
        results = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': len(valid_predictions),
            'final_prediction': final_prediction,
            'final_class_name': self.class_names[final_prediction] if final_prediction != -1 else "No prediction",
            'prediction_confidence': prediction_confidence,
            'average_confidence': avg_confidence,
            'average_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'real_time_factor': fps * avg_inference_time
        }
        
        return results
    
    def real_time_inference(self, camera_id: int = 0):
        """
        Real-time inference from camera.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        print("Starting real-time inference. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            pred_class, confidence, class_name = self.predict_single_frame(frame)
            
            # Draw prediction on frame
            if pred_class != -1:
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('RIGID Gait Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='RIGID Gait Recognition Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained RIGID model')
    parser.add_argument('--input', type=str, 
                        help='Input video file or camera ID (e.g., 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--frame_size', type=int, default=66,
                        help='Input frame size')
    parser.add_argument('--real_time', action='store_true',
                        help='Enable real-time inference from camera')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = RIGIDInference(args.model_path, args.frame_size)
    
    if args.real_time:
        # Real-time inference from camera
        camera_id = int(args.input) if args.input else 0
        inference.real_time_inference(camera_id)
    else:
        # Video file inference
        if not args.input:
            raise ValueError("Input video path is required for file inference")
        
        results = inference.predict_video(args.input, args.output)
        
        # Print results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        print(f"Video: {results['video_path']}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Processed frames: {results['processed_frames']}")
        print(f"Final prediction: {results['final_class_name']}")
        print(f"Prediction confidence: {results['prediction_confidence']:.3f}")
        print(f"Average confidence: {results['average_confidence']:.3f}")
        print(f"Average inference time: {results['average_inference_time_ms']:.2f} ms")
        print(f"Real-time factor: {results['real_time_factor']:.2f}")
        
        if args.output:
            print(f"Output video saved to: {args.output}")


if __name__ == "__main__":
    main()
