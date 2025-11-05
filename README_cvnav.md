# AuAir Flight Path Visualizer - cvnav Environment Setup

## Environment Information
- **Environment Name**: `cvnav`
- **Python Version**: 3.10.19
- **Purpose**: Computer Vision and Navigation analysis, specifically for flight path visualization

## Quick Start

### 1. Activate the Environment
```bash
conda activate cvnav
```

### 2. Verify Environment Setup
```bash
python verify_environment.py
```

### 3. Run Flight Path Visualizer
```bash
# Use default directory (data/annotations/annotation_files)
python flight_path_visualizer.py

# Use custom directory
python flight_path_visualizer.py /path/to/frame/files/

# Generate 2D analysis plots as well
python flight_path_visualizer.py --2d /path/to/frame/files/
```

## Generated Output Files
When you run the flight path visualizer, it creates:
- `flight_path_3d.png` - Static 3D matplotlib plot
- `flight_path_interactive_3d.html` - Interactive 3D Plotly plot (mouse rotatable)
- `flight_analysis_2d.png` - 2D analysis plots (if --2d flag used)

## Core Dependencies (Verified Working)
- ✅ **matplotlib** 3.10.7 - Static 3D plotting
- ✅ **numpy** 2.2.6 - Numerical operations
- ✅ **plotly** 6.3.1 - Interactive 3D HTML plots
- ✅ **pandas** 2.3.2 - Data manipulation
- ✅ **scipy** 1.15.3 - Scientific computing

## Deep Learning Framework
- ✅ **torch** 2.9.0+cu128 - PyTorch with CUDA support
- ✅ **torchvision** 0.24.0+cu128 - Computer vision utilities
- ✅ **torchaudio** 2.9.0+cu128 - Audio processing
- ✅ **torchmetrics** 1.8.2 - ML metrics

## Computer Vision
- ✅ **opencv-python** 4.12.0 - Computer vision operations
- ✅ **Pillow** 12.0.0 - Image processing

## Environment Reproduction
If you need to recreate this environment on another system:

### Option 1: Using environment.yml (Recommended)
```bash
conda env create -f environment.yml
conda activate cvnav
```

### Option 2: Using requirements.txt
```bash
conda create -n cvnav python=3.10
conda activate cvnav
pip install -r requirements.txt
```

## Troubleshooting
- Always make sure `conda activate cvnav` is run before executing scripts
- If packages are missing, run `python verify_environment.py` to check status
- All versions in requirements.txt are tested and compatible with Python 3.10+

## Usage Examples

### Basic Flight Path Visualization
```bash
conda activate cvnav
python flight_path_visualizer.py
```

### Full Analysis with 2D Plots
```bash
conda activate cvnav
python flight_path_visualizer.py --2d ./data/annotations/annotation_files/
```

The interactive HTML plot can be opened in any web browser for 3D exploration with mouse controls!