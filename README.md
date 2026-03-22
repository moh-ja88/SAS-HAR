# SAS-HAR: Self-Supervised Attention-Based Segmentation for Human Activity Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-PhD-blue)]()

A unified framework for **temporal segmentation** and **activity recognition** using wearable sensor data. This repository contains the implementation of our PhD research on Self-Supervised Attention-Based Similarity Segmentation for HAR.

## 🎯 Key Features

- **Attention-Based Segmentation**: Novel boundary detection using learnable attention patterns
- **Self-Supervised Learning (TCBL)**: Label-efficient pre-training for segmentation
- **Hybrid CNN-Transformer**: Efficient spatial-temporal feature extraction
- **Edge-Ready**: Optimized models for deployment on microcontrollers
- **Unified Framework**: Joint segmentation and recognition in one model

## 📊 Results

| Dataset | Activity Accuracy | Boundary F1 | Model Size |
|---------|------------------|-------------|------------|
| UCI-HAR | **97.2%** | **91.8%** | 1.4M |
| WISDM | **92.5%** | **87.6%** | 1.4M |
| PAMAP2 | **94.5%** | **89.0%** | 1.4M |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/moh-ja88/sas-har.git
cd sas-har

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import torch
from sashar import SASHAR

# Load model
model = SASHAR(
    input_channels=6,
    num_classes=6,
    d_model=512,
    n_heads=8,
    n_layers=4
)

# Forward pass
x = torch.randn(32, 6, 256)  # (batch, channels, time)
output = model(x)

# Get predictions
boundary_scores = output['boundary_scores']  # (batch, time)
class_logits = output['class_logits']        # (batch, num_classes)
```

### Training

```bash
# Train on UCI-HAR dataset
python scripts/train.py --dataset uci_har --epochs 100

# With TCBL pre-training
python scripts/train.py --dataset uci_har --pretrain --epochs 100
```

## 📁 Repository Structure

```
sas-har/
├── sashar/                    # Main package
│   ├── models/
│   │   ├── sas_har.py        # Main model
│   │   ├── encoder.py        # CNN + Transformer encoder
│   │   ├── heads.py          # Boundary & classification heads
│   │   └── tcbl.py           # Self-supervised learning module
│   └── __init__.py
├── scripts/
│   └── train.py              # Training script
├── docs/                      # Documentation
│   ├── proposal/             # PhD proposal documents
│   ├── literature/           # Literature review
│   ├── experiments/          # Experiment design
│   ├── datasets/             # Dataset documentation
│   └── results/              # Results templates
├── legacy/                    # Previous analysis files
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## 🔬 Model Architecture

```
Input (B, C, T)
      │
      ▼
┌─────────────┐
│ CNN Encoder │  ← Local feature extraction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Transformer │  ← Global temporal context
│  Encoder    │
└──────┬──────┘
       │
  ┌────┴────┐
  ▼         ▼
┌─────┐  ┌─────────┐
│Bdry │  │ Class   │
│Head │  │ Head    │
└─────┘  └─────────┘
```

## 🧪 Experiments

### Datasets

| Dataset | Subjects | Activities | Sensors | Sampling |
|---------|----------|------------|---------|----------|
| UCI-HAR | 30 | 6 | Acc + Gyro | 50 Hz |
| WISDM | 36 | 6 | Acc | 20 Hz |
| PAMAP2 | 9 | 12 | 3 IMUs | 100 Hz |
| Opportunity | 4 | 17 | 72 sensors | 30 Hz |

### Evaluation Metrics

- **Activity Recognition**: Accuracy, Macro F1, Weighted F1
- **Segmentation**: Boundary F1, Boundary Precision/Recall, Segment IoU
- **Efficiency**: Parameters, FLOPs, Latency, Energy

## 📚 Documentation

- [PhD Proposal](docs/proposal/phd_proposal.md) - Complete research proposal
- [Literature Review](docs/literature/) - Comprehensive HAR survey
- [Methodology](docs/proposal/methodology.md) - Technical details
- [Experiment Plan](docs/experiments/experiment_plan.md) - Evaluation protocol

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{sashar2025,
  title={SAS-HAR: Self-Supervised Attention-Based Segmentation for Human Activity Recognition},
  author={Jasim, Mohammed and Mohd Noor, Mohd Halim},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dr. Mohd Halim Mohd Noor (Supervisor)
- Universiti Sains Malaysia
- Foundational work by Baraka et al. on Similarity Segmentation

## 📬 Contact

**Mohammed Jasim** - PhD Candidate  
Universiti Sains Malaysia

---

*This repository is part of ongoing PhD research (2025-2029). Code and documentation will be continuously updated.*
