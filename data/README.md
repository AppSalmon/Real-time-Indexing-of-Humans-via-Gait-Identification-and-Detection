# Data Directory

This directory contains datasets and processed data for RIGID gait recognition.

## Dataset Information

### DMGait Dataset (Primary Training Dataset)
- **Source**: https://www.nature.com/articles/s41597-023-02161-8#citeas
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
- **Source**: [Kaggle - RealWorld Gait Dataset](https://www.kaggle.com/datasets/salmon1/realworld-gait)
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
- **Source**: [Kaggle - RealWorld Gait Sequence Dataset](https://www.kaggle.com/datasets/caophankhnhduy/realworldgait)
- **Participants**: 13 people (RealWorld-v1 + 7 additional participants)
- **Total videos**: >300 clips
- **Purpose**: Multi-condition testing to ensure diverse recording conditions
- **Characteristics**:
  - Combines RealWorld-v1 (6 participants) + 7 additional participants
  - Multiple recording conditions
  - Various environments and scenarios
  - Comprehensive testing dataset
  - Used for evaluating model robustness across different conditions

### RealWorld Dataset Naming Convention
Each video follows a strict naming pattern: `P{subject}_B{background}_C{clothing}_A{angle}_D{direction}_R{repetition}.mp4`

| Component | Values | Description |
|-----------|--------|-------------|
| **P** | `P1` → `P7` | Subject ID (7 participants) |
| **B** | `B1`, `B2` | Background |
| | `B1`: Bicycle yard (outdoor, natural light) |
| | `B2`: Laboratory (indoor, artificial light) |
| **C** | `N`, `C`, `B` | Clothing |
| | `N`: None (no coat, no backpack) |
| | `C`: Coat only |
| | `B`: Coat & Backpack |
| **A** | `0`, `90` | View angle |
| | `0`: Front or rear view |
| | `90`: Side view (perpendicular to walking direction) |
| **D** | `L`, `R` | Walking direction |
| | `L`: Left-to-right |
| | `R`: Right-to-left |
| **R** | `1`, `2` | Repetition |
| | `1`: First recording |
| | `2`: Repeated recording of the same setup |

**Example**: `P2_B1_B_0_R_2.mp4` = Subject 2, background B1 (bicycle yard), wearing coat & backpack, 0° camera view, walking right-to-left, repetition 2.

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
