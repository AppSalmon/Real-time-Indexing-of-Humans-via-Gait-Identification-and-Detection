# RIGID: Real-time Indexing of Humans via Gait Identification and Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight and efficient framework for real-time human gait recognition using RGB video data, achieving state-of-the-art performance with minimal computational requirements.

## 🚀 Key Features

- **Real-time Performance**: 3.02 ms inference time per frame
- **Lightweight Model**: Only 578K parameters
- **High Accuracy**: 96.42% on DMGait dataset (64 participants)
- **RGB-only Input**: No need for silhouette or depth data
- **Tri-channel Representation**: Combines coordinate maps, distance matrices, and grayscale frames
- **ConvNeXt V2 Backbone**: Modern CNN architecture optimized for efficiency

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 96.42% |
| Parameters | 578K |
| Inference Time | 3.02 ms/frame |
| Model Size | < 2.5 MB |
| Input Size | 66×66×3 |
| ConvNeXt Blocks | (1,1,1) with dims (50,100,200) |

## 🏗️ Architecture

RIGID employs a novel tri-channel frame representation:

1. **Coordinate Map (P)**: Joint positions from MediaPipe pose landmarks
2. **Distance Matrix (J)**: Inter-joint Euclidean distances
3. **Grayscale Frame (D)**: Original video frame

These channels are stacked into a 3D tensor and processed through a ConvNeXt V2 backbone for efficient spatio-temporal feature extraction.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- MediaPipe
- OpenCV

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rigid-gait-recognition.git
cd rigid-gait-recognition

# Install dependencies
pip install -r requirements.txt

# Download datasets (see data/ directory)
```

## 📁 Project Structure

```
rigid-gait-recognition/
├── src/
│   ├── models/
│   │   ├── convnext_v2.py      # ConvNeXt V2 backbone
│   │   └── rigid_model.py      # Main RIGID model
│   ├── data/
│   │   ├── preprocessing.py    # Data preprocessing utilities
│   │   └── feature_extraction.py # MediaPipe feature extraction
│   └── utils/
│       ├── visualization.py    # Plotting and visualization
│       └── metrics.py         # Evaluation metrics
├── notebooks/
│   ├── data_analysis.ipynb    # Dataset exploration
│   └── model_analysis.ipynb   # Model performance analysis
├── tests/
│   ├── test_models.py         # Unit tests for models
│   └── test_data.py           # Unit tests for data processing
├── config/
│   └── default_config.yaml    # Default hyperparameters
├── data/                      # Dataset directory
├── models/                    # Trained model checkpoints
└── docs/                      # Documentation
```

## 📊 Datasets

RIGID has been evaluated on multiple datasets:

### Primary Dataset (Training & Main Evaluation)
- **DMGait Dataset**: Large-scale diverse gait dataset with 64 participants
  - **Participants**: 64 people (37 male, 27 female, age 18-65)
  - **Total walks**: 1,560 (24 walks per person)
  - **Total videos**: 3,120 (2 cameras per walk)
  - **Total frames**: ~748,800 frames
  - **Keypoints**: ~56.16M keypoints (75 per frame using OpenPose)
  - **Views**: 8 angles (0°-315°, every 45°)
  - **Environments**: Indoor and outdoor
  - **Clothing variations**: 2 different sets per person
  - **Sensor data**: Goniometer data at 84 Hz
  
  **Citation**: Topham, L.K., Khan, W., Al-Jumeily, D. et al. A diverse and multi-modal gait dataset of indoor and outdoor walks acquired using multiple cameras and sensors. Sci Data 10, 320 (2023). https://doi.org/10.1038/s41597-023-02161-8
  
  **Dataset Link**: [DMGait Dataset](https://www.nature.com/articles/s41597-023-02161-8#citeas)

### Real-world Testing Datasets
- **RealWorld-v1**: Limited data testing (6 subjects)
- **RealWorld-v2**: Multi-environment testing with occlusions (13 subjects)

**RealWorld Dataset Links:**
- [RealWorld Gait Dataset](https://www.kaggle.com/datasets/salmon1/realworld-gait)
- [RealWorld-v2 Sequence Dataset](https://www.kaggle.com/datasets/caophankhnhduy/realworldgait)

## 🔬 Ablation Studies

Key findings from our experiments:

- **Optimal ConvNeXt blocks**: 3-6 blocks provide best performance
- **Dense layer size**: 128 dimensions optimal
- **Dropout rate**: 0.25 provides best regularization
- **Feature importance**: Tri-channel representation crucial for performance

## 📈 Results Comparison

| Method | Accuracy | Parameters | Inference (ms) |
|--------|----------|------------|----------------|
| GaitSet | 91.62% | 2.37M | - |
| GaitPart | 76.76% | 1.90M | - |
| Skeleton-LSTM | 93.01% | 0.40M | - |
| **RIGID (ours)** | **96.42%** | **0.58M** | **3.02** |

## 🎯 Use Cases

- **Security Systems**: Access control and surveillance
- **Healthcare**: Gait analysis and rehabilitation monitoring
- **IoT Applications**: Edge device deployment
- **Privacy-preserving Authentication**: When facial recognition is not suitable

## 🔧 Configuration

Key hyperparameters can be configured in `config/default_config.yaml`:

```yaml
model:
  convnext_blocks: 3  # Fixed: (1,1,1) depths with (50,100,200) dims
  dense_dim: 128
  dropout_rate: 0.25
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 20

data:
  frame_size: 66  # RIGID specific input size
  sequence_length: 30
  num_keypoints: 33
```

## 📝 Citation

<!-- If you use RIGID in your research, please cite: -->

<!-- ```bibtex
@article{rigid2025,
  title={RIGID: Real-time Indexing of Humans via Gait Identification and Detection},
  author={Dao-Xuan, H.-T. and Cao-Phan, K.-D. and Nguyen, V.-L.},
  journal={Multimedia Tools and Applications},
  year={2025}
}
``` -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Supported by NSTC Taiwan Grant 112-2221-E-194-017-MY3
- Thanks to MediaPipe team for pose estimation tools
- DMGait dataset contributors for providing evaluation benchmarks

---

**Note**: This implementation focuses on the core RIGID framework. For full experimental details and dataset preparation, refer to the original paper.
