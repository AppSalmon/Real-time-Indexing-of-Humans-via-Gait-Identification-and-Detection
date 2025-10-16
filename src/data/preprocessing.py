"""
Data Preprocessing Module for RIGID Gait Recognition

This module handles the preprocessing pipeline for gait recognition data,
including video loading, pose extraction, and tri-channel feature generation.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Union
import os
from pathlib import Path


class PoseExtractor:
    """MediaPipe pose extraction for gait recognition."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Key joint indices for gait analysis (33 total)
        self.joint_indices = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
            'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
            'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
            'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def extract_pose_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Array of shape (33, 3) with (x, y, visibility) for each landmark
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.visibility])
            return np.array(landmarks)
        
        return None
    
    def extract_poses_from_video(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract pose landmarks from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            List of pose landmark arrays
        """
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_count >= max_frames:
                break
                
            pose = self.extract_pose_from_frame(frame)
            if pose is not None:
                poses.append(pose)
            
            frame_count += 1
        
        cap.release()
        return poses
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


class TriChannelGenerator:
    """Generate tri-channel features for RIGID model."""
    
    def __init__(self, frame_size: int = 66):
        self.frame_size = frame_size
        self.num_joints = 33
    
    def create_coordinate_map(self, pose: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create coordinate map from pose landmarks.
        
        Args:
            pose: Pose landmarks (33, 3)
            frame_shape: Original frame shape (height, width)
            
        Returns:
            Coordinate map of shape (frame_size, frame_size, 1)
        """
        coord_map = np.zeros((self.frame_size, self.frame_size, 1))
        
        if pose is None:
            return coord_map
        
        height, width = frame_shape[:2]
        
        for joint in pose:
            x, y, visibility = joint
            
            # Skip low visibility joints
            if visibility < 0.5:
                continue
            
            # Convert to pixel coordinates
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            
            # Scale to frame_size
            scaled_x = int(pixel_x * self.frame_size / width)
            scaled_y = int(pixel_y * self.frame_size / height)
            
            # Ensure within bounds
            scaled_x = max(0, min(scaled_x, self.frame_size - 1))
            scaled_y = max(0, min(scaled_y, self.frame_size - 1))
            
            # Create heatmap around joint
            coord_map[scaled_y, scaled_x, 0] = visibility
        
        return coord_map
    
    def create_distance_matrix(self, pose: np.ndarray) -> np.ndarray:
        """
        Create distance matrix from pose landmarks.
        
        Args:
            pose: Pose landmarks (33, 3)
            
        Returns:
            Distance matrix of shape (frame_size, frame_size, 1)
        """
        dist_matrix = np.zeros((self.frame_size, self.frame_size, 1))
        
        if pose is None:
            return dist_matrix
        
        # Calculate pairwise distances between joints
        joint_coords = pose[:, :2]  # x, y coordinates
        
        # Create distance map
        for i in range(self.num_joints):
            for j in range(i + 1, self.num_joints):
                if pose[i, 2] > 0.5 and pose[j, 2] > 0.5:  # Both joints visible
                    # Calculate distance
                    dist = np.linalg.norm(joint_coords[i] - joint_coords[j])
                    
                    # Normalize distance (assuming max distance is sqrt(2))
                    normalized_dist = min(dist / np.sqrt(2), 1.0)
                    
                    # Map to matrix coordinates
                    x_coord = int(i * self.frame_size / self.num_joints)
                    y_coord = int(j * self.frame_size / self.num_joints)
                    
                    dist_matrix[y_coord, x_coord, 0] = normalized_dist
                    dist_matrix[x_coord, y_coord, 0] = normalized_dist  # Symmetric
        
        return dist_matrix
    
    def create_grayscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Create grayscale frame and resize.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Grayscale frame of shape (frame_size, frame_size, 1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to frame_size
        gray_resized = cv2.resize(gray, (self.frame_size, self.frame_size))
        
        # Normalize to [0, 1]
        gray_normalized = gray_resized.astype(np.float32) / 255.0
        
        # Add channel dimension
        return np.expand_dims(gray_normalized, axis=-1)
    
    def generate_tri_channel_features(
        self, 
        pose: np.ndarray, 
        frame: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate tri-channel features for a single frame.
        
        Args:
            pose: Pose landmarks (33, 3)
            frame: Original frame
            
        Returns:
            Dictionary with three channels:
            - 'coordinate_maps': Coordinate map
            - 'distance_matrices': Distance matrix  
            - 'grayscale_frames': Grayscale frame
        """
        frame_shape = frame.shape
        
        coord_map = self.create_coordinate_map(pose, frame_shape)
        dist_matrix = self.create_distance_matrix(pose)
        gray_frame = self.create_grayscale_frame(frame)
        
        return {
            'coordinate_maps': coord_map,
            'distance_matrices': dist_matrix,
            'grayscale_frames': gray_frame
        }


class GaitDataProcessor:
    """Main data processor for gait recognition pipeline."""
    
    def __init__(self, frame_size: int = 66):
        self.frame_size = frame_size
        self.pose_extractor = PoseExtractor()
        self.tri_channel_generator = TriChannelGenerator(frame_size)
    
    def process_video_file(
        self, 
        video_path: str, 
        label: int,
        max_frames: int = None
    ) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Process a video file and extract tri-channel features.
        
        Args:
            video_path: Path to video file
            label: Class label for the video
            max_frames: Maximum number of frames to process
            
        Returns:
            Tuple of (tri_channel_features_list, labels_list)
        """
        # Extract poses from video
        poses = self.pose_extractor.extract_poses_from_video(video_path, max_frames)
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_count >= max_frames:
                break
                
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # Generate tri-channel features
        features_list = []
        labels_list = []
        
        min_length = min(len(poses), len(frames))
        
        for i in range(min_length):
            if poses[i] is not None:  # Only process frames with valid poses
                features = self.tri_channel_generator.generate_tri_channel_features(
                    poses[i], frames[i]
                )
                features_list.append(features)
                labels_list.append(label)
        
        return features_list, labels_list
    
    def process_dataset(
        self, 
        dataset_path: str,
        max_frames_per_video: int = 30,
        max_videos_per_class: int = None
    ) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Process entire dataset directory.
        
        Args:
            dataset_path: Path to dataset directory
            max_frames_per_video: Maximum frames to process per video
            max_videos_per_class: Maximum videos per class
            
        Returns:
            Tuple of (all_features, all_labels)
        """
        all_features = []
        all_labels = []
        
        dataset_path = Path(dataset_path)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(dataset_path.rglob(f'*{ext}'))
        
        # Group by class (assuming folder structure)
        class_videos = {}
        for video_file in video_files:
            class_name = video_file.parent.name
            if class_name not in class_videos:
                class_videos[class_name] = []
            class_videos[class_name].append(video_file)
        
        # Process each class
        for class_idx, (class_name, videos) in enumerate(class_videos.items()):
            print(f"Processing class {class_name} ({class_idx}) with {len(videos)} videos...")
            
            if max_videos_per_class:
                videos = videos[:max_videos_per_class]
            
            for video_path in videos:
                try:
                    features, labels = self.process_video_file(
                        str(video_path), 
                        class_idx, 
                        max_frames_per_video
                    )
                    all_features.extend(features)
                    all_labels.extend(labels)
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue
        
        return all_features, all_labels
    
    def save_processed_data(
        self, 
        features: List[Dict[str, np.ndarray]], 
        labels: List[int],
        save_path: str
    ):
        """
        Save processed data to file.
        
        Args:
            features: List of tri-channel features
            labels: List of labels
            save_path: Path to save processed data
        """
        # Convert to numpy arrays
        coord_maps = np.array([f['coordinate_maps'] for f in features])
        dist_matrices = np.array([f['distance_matrices'] for f in features])
        gray_frames = np.array([f['grayscale_frames'] for f in features])
        labels_array = np.array(labels)
        
        # Save as compressed numpy file
        np.savez_compressed(
            save_path,
            coordinate_maps=coord_maps,
            distance_matrices=dist_matrices,
            grayscale_frames=gray_frames,
            labels=labels_array
        )
        
        print(f"Processed data saved to {save_path}")
        print(f"Shape: {coord_maps.shape}, Labels: {len(labels_array)}")


def load_processed_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load processed data from file.
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        Tuple of (coordinate_maps, distance_matrices, grayscale_frames, labels)
    """
    data = np.load(data_path)
    
    return (
        data['coordinate_maps'],
        data['distance_matrices'], 
        data['grayscale_frames'],
        data['labels']
    )


if __name__ == "__main__":
    # Test the data processor
    processor = GaitDataProcessor(frame_size=224)
    
    # Test with a sample video (replace with actual path)
    # features, labels = processor.process_video_file("sample_video.mp4", 0)
    # print(f"Processed {len(features)} frames")
    
    print("Data preprocessing module loaded successfully!")
