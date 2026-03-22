# SAS-HAR Code Structure

This directory contains all implementation code for the SAS-HAR framework.

## Directory Structure

```
07_code/
├── sashar/                    # Main package
│   ├── __init__.py
│   ├── models/                # Model definitions
│   │   ├── __init__.py
│   │   ├── sas_har.py        # Main model
│   │   ├── encoder.py        # CNN + Transformer encoder
│   │   ├── heads.py          # Task-specific heads
│   │   └── tcbl.py           # Self-supervised learning
│   │
│   ├── training/             # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py        # Main trainer class
│   │   ├── losses.py         # Loss functions
│   │   └── optimizer.py      # Optimizer utilities
│   │
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   ├── datasets.py       # Dataset classes
│   │   ├── preprocessing.py  # Data preprocessing
│   │   └── augmentation.py   # Data augmentation
│   │
│   ├── evaluation/           # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py        # Metric computations
│   │   └── visualization.py  # Result visualization
│   │
│   └── deployment/           # Edge deployment
│       ├── __init__.py
│       ├── quantization.py   # Quantization utilities
│       ├── distillation.py   # Knowledge distillation
│       └── export.py         # Model export utilities
│
├── baselines/                # Baseline implementations
│   ├── __init__.py
│   ├── fixed_window.py
│   ├── adaptive_window.py
│   ├── statistical_similarity.py
│   └── deep_similarity.py
│
├── evaluation/               # Evaluation scripts
│   ├── __init__.py
│   ├── evaluate_baselines.py
│   └── ablation_studies.py
│
├── training/                 # Training scripts
│   ├── __init__.py
│   ├── train_self_supervised.py
│   ├── train_supervised.py
│   └── train_distillation.py
│
├── configs/                  # Configuration files
│   ├── default.yaml
│   ├── wisdm.yaml
│   ├── uci_har.yaml
│   └── pamap2.yaml
│
├── scripts/                  # Utility scripts
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── run_experiments.py
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_comparison.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                    # Unit tests
│   ├── test_models.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── requirements.txt          # Python dependencies
└── setup.py                  # Package setup
```

## Key Modules

### 1. Models (`sashar/models/`)

| File | Purpose |
|------|---------|
| `sas_har.py` | Main SAS-HAR model class |
| `encoder.py` | CNN and Transformer encoder modules |
| `heads.py` | Boundary detection and classification heads |
| `tcbl.py` | Self-supervised learning components |

### 2. Training (`sashar/training/`)

| File | Purpose |
|------|---------|
| `trainer.py` | Main training loop and utilities |
| `losses.py` | Loss functions for all training modes |
| `optimizer.py` | Optimizer configuration and scheduling |

### 3. Data (`sashar/data/`)

| File | Purpose |
|------|---------|
| `datasets.py` | Dataset classes for WISDM, UCI-HAR, etc. |
| `preprocessing.py` | Data normalization and filtering |
| `augmentation.py` | Data augmentation techniques |

### 4. Evaluation (`sashar/evaluation/`)

| File | Purpose |
|------|---------|
| `metrics.py` | Accuracy, F1, boundary metrics |
| `visualization.py` | Confusion matrices, attention visualization |

### 5. Deployment (`sashar/deployment/`)

| File | Purpose |
|------|---------|
| `quantization.py` | INT8 quantization utilities |
| `distillation.py` | Knowledge distillation training |
| `export.py` | Export to ONNX, TFLite |

## Usage Examples

### Training

```python
from sashar import SASHAR, SelfSupervisedTrainer, SupervisedTrainer

# Phase 1: Self-supervised pre-training
model = SASHAR(input_channels=6, num_classes=6)
trainer = SelfSupervisedTrainer(model)
trainer.train(unlabeled_data, epochs=100)

# Phase 2: Supervised fine-tuning
trainer = SupervisedTrainer(model)
trainer.train(labeled_data, epochs=50)
```

### Evaluation

```python
from sashar.evaluation import evaluate_segmentation, evaluate_classification

# Evaluate segmentation
boundary_f1 = evaluate_segmentation(model, test_data)

# Evaluate classification
accuracy, f1 = evaluate_classification(model, test_data)
```

### Deployment

```python
from sashar.deployment import distill_model, quantize_model

# Knowledge distillation
student_model = distill_model(teacher_model, train_data)

# Quantization
quantized_model = quantize_model(student_model)
```

## Running Experiments

```bash
# Self-supervised pre-training
python training/train_self_supervised.py --config configs/default.yaml

# Supervised fine-tuning
python training/train_supervised.py --config configs/wisdm.yaml

# Evaluate baselines
python evaluation/evaluate_baselines.py --dataset wisdm

# Run ablation studies
python evaluation/ablation_studies.py --config configs/default.yaml
```

## Configuration

Example `configs/default.yaml`:

```yaml
model:
  input_channels: 6
  num_classes: 6
  hidden_dim: 256
  num_heads: 4
  num_layers: 3
  use_transition_module: true

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  weight_decay: 0.0001

self_supervised:
  temperature: 0.1
  mask_ratio: 0.15
  lambda_tc: 1.0
  lambda_cp: 0.5
  lambda_mt: 0.3

deployment:
  target_params: 25000
  quantize: true
  export_format: "tflite"
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=sashar --cov-report=html
```

---

*Last Updated: March 2026*
