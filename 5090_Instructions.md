# NeuS2 RTX 5090 Setup Instructions

This document outlines the complete setup process for running NeuS2 on an RTX 5090 (compute capability 12.0) with CUDA 12.8.

## System Requirements

- **GPU**: NVIDIA RTX 5090 (compute capability 12.0)
- **CUDA**: 12.8+
- **OS**: Linux (tested on Ubuntu 22.04+)
- **Python**: 3.9 (recommended for compatibility)

## Initial Problem

The original NeuS2 build was targeting compute capability 8.9 (Ada architecture), which caused:
1. Training to stall at iteration 183
2. Loss values getting stuck at the same values for multiple iterations
3. Poor gradient flow due to architecture mismatch

## Root Cause Identified

**CUDA Architecture Mismatch**: The original build was targeting compute capability 8.9 (Ada architecture) instead of 12.0 (Blackwell architecture). This caused training to stall at low iterations due to kernel incompatibility.

## Complete Setup Process

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n neus python=3.9
conda activate neus

# Install PyTorch nightly with CUDA 12.8 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install evaluation metrics
pip install torchmetrics==1.8.1 lpips==0.1.4

# Install other dependencies
pip install -r requirements.txt
```

### 2. Clone and Configure NeuS2

```bash
# Clone the repository
git clone --recursive https://github.com/19reborn/NeuS2
cd NeuS2

# Set CUDA architecture environment variables
export TCNN_CUDA_ARCHITECTURES=120
export CMAKE_CUDA_ARCHITECTURES=120

# Configure CMake for RTX 5090
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DNGP_BUILD_WITH_OPTIX=OFF \
  -DPython_EXECUTABLE=/path/to/your/conda/envs/neus/bin/python3.9 \
  -DPython_ROOT_DIR=/path/to/your/conda/envs/neus \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

### 3. Build the Project

```bash
# Build the main testbed
cmake --build build --config RelWithDebInfo -j

# Build the Python extension module
cmake --build build --target pyngp -j
```

### 4. Verify Build

```bash
# Check that the binary was built for sm_120
strings build/testbed | grep "sm_"

# Verify Python module was built for Python 3.9
ls -1 build/pyngp*.so
# Should show: pyngp.cpython-39-x86_64-linux-gnu.so
```

### 5. Test Training

```bash
# Set PYTHONPATH for Python imports
export PYTHONPATH=/path/to/NeuS2/build:$PYTHONPATH

# Test C++ binary
./build/testbed --scene /path/to/data/lego/transforms_train.json --no-gui

# Test Python script
python scripts/run.py --scene /path/to/data/lego/transforms_train.json --name lego_test --n_steps 200
```

## Configuration Files

### Configuration

The default `configs/nerf/base.json` works fine with the sm_120 build. No configuration changes are needed - the architecture fix resolves the training issues.

## Troubleshooting

### Training Stalls at Low Iterations

**Symptoms**: Loss gets stuck at the same value for multiple iterations
**Cause**: Wrong CUDA architecture (sm_89 instead of sm_120)

**Solution**: Rebuild with `TCNN_CUDA_ARCHITECTURES=120`

### Python Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'pyngp'`
**Causes**: 
- Wrong Python version in build
- PYTHONPATH not set correctly

**Solutions**:
1. Ensure CMake finds Python 3.9
2. Set `PYTHONPATH=/path/to/NeuS2/build:$PYTHONPATH`
3. Verify `.so` file matches your Python version

### Build Failures

**Symptoms**: CMake configuration errors
**Causes**:
- Missing Python development headers
- CUDA architecture not set correctly

**Solutions**:
1. Install Python dev packages: `conda install python=3.9`
2. Set environment variables: `export TCNN_CUDA_ARCHITECTURES=120`
3. Use policy flag: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`

## Performance Notes

- **Training Speed**: RTX 5090 with sm_120 build is significantly faster than sm_89
- **Memory Usage**: HashGrid encoding uses ~10.5M parameters
- **Convergence**: Training progresses normally with the default configuration

## File Locations

- **Binary**: `build/testbed`
- **Python Module**: `build/pyngp.cpython-39-x86_64-linux-gnu.so`
- **Configs**: `configs/nerf/`
- **Scripts**: `scripts/run.py`

## Environment Variables Summary

```bash
export TCNN_CUDA_ARCHITECTURES=120
export CMAKE_CUDA_ARCHITECTURES=120
export PYTHONPATH=/path/to/NeuS2/build:$PYTHONPATH
```

## Verification Commands

```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Verify CUDA version
nvcc --version

# Test training progression
./build/testbed --scene /path/to/data --no-gui | grep "iteration="

# Check Python module
python -c "import pyngp as ngp; print('Success')"
```

## Notes

- The original issue was **not** PyTorch3D compatibility
- Training stalls were caused by architecture mismatch only
- The default configuration works fine once built for sm_120
- Always verify the build targets sm_120 for Blackwell architecture 