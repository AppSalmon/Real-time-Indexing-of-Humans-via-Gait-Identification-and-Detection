# Data Directory

This directory contains datasets and processed data for RIGID gait recognition.

## Directory Structure

```
data/
├── raw/                    # Raw video data
│   ├── dmgait/            # DMGait dataset videos (64 participants)
│   ├── realworld-v1/      # RealWorld-v1 dataset videos (6 participants)
│   └── realworld-v2/      # RealWorld-v2 dataset videos (13 participants)
├── processed/             # Preprocessed data
│   ├── dmgait_dataset.npz # Processed DMGait features
│   ├── realworld_v1_dataset.npz # Processed RealWorld-v1 features
│   └── realworld_v2_dataset.npz # Processed RealWorld-v2 features
└── cache/                 # Temporary cache files
```

## Dataset Information

### DMGait Dataset (Primary Training Dataset)
- **Source**: [Kaggle - Large Gait Dataset](https://www.kaggle.com/datasets/salmon1/large-gait-dataset/data)
- **Participants**: 64 people (37 male, 27 female, age 18-65)
- **Total videos**: 3,120 (2 cameras per walk)
- **Total walks**: 1,560 (24 walks per person)
- **Views**: 8 angles (0°-315°, every 45°)
- **Environments**: Indoor and outdoor
- **Clothing variations**: 2 different sets per person
- **Sensor data**: Goniometer data at 84 Hz
- **Keypoints**: ~56.16M keypoints (75 per frame using OpenPose)
- **Total frames**: ~748,800 frames
- **Duration**: ~415 minutes total video

**Citation**: Topham, L.K., Khan, W., Al-Jumeily, D. et al. A diverse and multi-modal gait dataset of indoor and outdoor walks acquired using multiple cameras and sensors. Sci Data 10, 320 (2023). https://doi.org/10.1038/s41597-023-02161-8

### RealWorld-v1 Dataset (Limited Data Testing)
- **Participants**: 6 people (folders: ha, lukas, nguyen, put, tuan, duy)
- **Recording setup**: Single angle view
- **Total videos**: ~95 videos (6 participants)
- **Purpose**: Limited data testing to ensure very little training data
- **Characteristics**: 
  - Controlled environment
  - Single view angle
  - Minimal data per participant
  - Used for testing model performance with limited training data

### RealWorld-v2 Dataset (Multi-condition Testing)
- **Participants**: 13 people (RealWorld-v1 + 7 additional participants)
- **Total videos**: >300 clips
- **Purpose**: Multi-condition testing to ensure diverse recording conditions
- **Characteristics**:
  - Combines RealWorld-v1 (6 participants) + 7 additional participants
  - Multiple recording conditions
  - Various environments and scenarios
  - Comprehensive testing dataset
  - Used for evaluating model robustness across different conditions

## Dataset Usage

### Training (Primary)
- **DMGait Dataset**: Main dataset for training RIGID model
- 64 participants provide sufficient diversity for robust gait recognition
- Multiple angles and environments ensure good generalization

### Testing (Real-world Evaluation)
- **RealWorld-v1**: Test model performance with limited data
- **RealWorld-v2**: Test model robustness across diverse conditions
- Both datasets used for real-world validation of RIGID performance

## Data Processing

### Preprocessing Pipeline
1. **Video Loading**: Load MP4 video files
2. **Pose Extraction**: Use MediaPipe to extract 33 keypoints per frame
3. **Tri-channel Generation**:
   - Coordinate maps (P): Joint positions
   - Distance matrices (J): Inter-joint distances
   - Grayscale frames (D): Original video frames
4. **Feature Stacking**: Combine into (66, 66, 3) tensor
5. **Data Saving**: Store as compressed .npz files

### Usage Instructions
1. Download datasets from provided links
2. Place raw videos in appropriate subdirectories under `raw/`
3. Run preprocessing scripts to generate `.npz` files in `processed/`
4. Use processed data for training and evaluation

## File Formats

### Raw Data
- **Video files**: MP4 format
- **Resolution**: Various (will be resized to 66x66 during preprocessing)
- **Frame rate**: 30 FPS (DMGait), variable (RealWorld)

### Processed Data
- **Format**: Compressed NumPy arrays (.npz)
- **Structure**:
  ```python
  {
      'coordinate_maps': (N, 66, 66, 1),    # Joint position maps
      'distance_matrices': (N, 66, 66, 1),  # Inter-joint distances
      'grayscale_frames': (N, 66, 66, 1),   # Grayscale video frames
      'labels': (N,)                        # Class labels
  }
  ```

## Notes

- Raw video files are not included in this repository due to size constraints
- Processed `.npz` files contain tri-channel features and labels ready for training
- Cache directory is for temporary processing files and can be safely deleted
- Always regenerate processed files when raw data changes
- Ensure proper file permissions when working with large datasets
