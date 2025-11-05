# AuAir Project Development Chat History

This file contains the development history and key decisions made during the creation of this drone navigation project.

## Project Overview

This is a computer vision-based drone navigation system that:
- Processes drone flight annotation files (frame_xxxxx.txt format)
- Implements 3D GPS visualization with coordinate conversion
- Trains neural networks for navigation prediction using multi-GPU setup
- Includes comprehensive data normalization for proper training convergence

## Key Development Milestones

### 1. Initial GPS Visualization (Flight Path Visualizer)
- Created `flight_path_visualizer.py` to read drone annotation files
- Implemented 3D plotting of latitude, longitude, altitude coordinates
- Added interactive Plotly visualizations with HTML export
- Processed 32,823 drone flight data points

### 2. Coordinate System Conversion
- Implemented WGS84 coordinate conversion functions in `utils.py`
- Added `deg_to_meters()` and `meters_to_deg()` functions
- Integrated coordinate conversion into visualization pipeline
- Used Earth radius = 6378137.0m for accurate conversions

### 3. Multi-GPU Training Setup
- Enhanced `train.py` with DataParallel support for 4x NVIDIA RTX A6000 GPUs
- Added automatic GPU detection and distributed training
- Implemented comprehensive CSV logging with timestamps
- Added validation split functionality for proper model evaluation

### 4. Critical Data Normalization Fixes
- **Major Issue Discovered**: Images and navigation data were not properly normalized
- Fixed image transform pipeline: ToTensor() → Normalize() (ImageNet standards)
- Implemented automatic navigation statistics computation from Denmark flight data
- Added normalization/denormalization utilities for proper neural network training
- Created validation scripts to ensure normalization correctness

### 5. Code Refactoring
- Consolidated all normalization utilities into `utils.py`
- Removed redundant `normalization_utils.py` file
- Updated all imports across the codebase
- Maintained backward compatibility and functionality

## Technical Specifications

### Hardware Environment
- 4x NVIDIA RTX A6000 GPUs (47.4 GB VRAM each)
- Python 3.10.19 in cvnav conda environment
- PyTorch 2.9.0 with CUDA support

### Dataset Characteristics
- Denmark flight coordinates: lat=56.2068°±0.0005°, lon=10.1886°±0.0004°
- Altitude range: 20,630m ± 6,917m
- Very tight GPS coordinate variation (±0.0005° lat/lon)

### Key Libraries
- matplotlib 3.10.7 for visualization
- plotly 6.3.1 for interactive plots
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## File Structure

```
config.py              # Configuration settings
dataset.py             # Data loading with proper normalization
models.py              # Neural network architectures
train.py               # Multi-GPU training script
utils.py               # Utilities (coordinates + normalization)
losses.py              # Loss functions
eval.py                # Evaluation scripts
test_normalization.py  # Validation for data normalization
launch_training.py     # Training launcher
train_distributed.py  # Distributed training alternative

scripts/
├── flight_path_visualizer.py  # GPS visualization system
├── flight_path_interactive_3d.html  # Interactive 3D plot output
└── json_section_viewer.py     # JSON data viewer

data/                  # Training data (excluded from git)
checkpoints/           # Model checkpoints (excluded from git)
logs/                  # Training logs (excluded from git)
```

## Critical Lessons Learned

1. **Data Normalization is Critical**: Proper normalization was essential for training convergence
2. **Multi-GPU Tensor Handling**: Required careful scalar extraction for loss computation
3. **Denmark Flight Data Characteristics**: Very tight GPS coordinates required specialized statistics
4. **Transform Pipeline Order**: ToTensor() must come before Normalize() in PyTorch

## Validation Status

All systems validated and working:
- ✅ GPS visualization with meter coordinate conversion
- ✅ Multi-GPU training setup with proper tensor handling
- ✅ Image normalization (ImageNet standards)
- ✅ Navigation data normalization with Denmark flight statistics
- ✅ Complete test suite passing

## Next Steps

Ready for production training:
1. Test on 10% dataset subset (cfg.dataset_fraction = 0.1)
2. Scale to full dataset training (cfg.dataset_fraction = 1.0)
3. Monitor training via CSV logs in logs/ directory

---
*Development completed: November 5, 2025*
*Environment: cvnav conda environment with PyTorch 2.9.0*