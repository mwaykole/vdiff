# Installation

## Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.1+ (for GPU inference)
- At least 16GB GPU memory for 8B models

## Installation Methods

### From Source (Recommended)

```bash
git clone https://github.com/your-org/vdiff.git
cd vdiff
pip install -r requirements.txt
pip install -e .
```

### From PyPI

```bash
pip install vdiff
```

### Using Docker

```bash
docker pull quay.io/your-org/vdiff:latest
```

## Verify Installation

```bash
python -c "import vdiff; print(vdiff.__version__)"
```

## GPU Setup

Ensure CUDA is properly installed:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
