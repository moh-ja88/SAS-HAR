# PhD Research Repository Improvement Plan

**Generated:** March 2026  
**Status:** Analysis Complete  
**Target:** Publication-Ready Research Framework

---

## Executive Summary

This document outlines a comprehensive improvement plan to transform the existing PhD research repository into a **fully reproducible, publication-ready research framework** capable of producing multiple academic publications at venues such as UbiComp, IEEE Sensors, IEEE IoT Journal, and Pattern Recognition.

### Current Assessment

| Dimension | Score | Status |
|-----------|-------|--------|
| **Code Quality** | 8/10 | Good structure, needs tests |
| **Documentation** | 8/10 | Comprehensive, needs figures |
| **Reproducibility** | 5/10 | Missing tests, CI/CD |
| **Publication Readiness** | 6/10 | Outlines exist, manuscripts needed |
| **Dataset Support** | 7/10 | 4/5 major datasets supported |

### Estimated Effort

- **High Priority Tasks:** ~80 hours
- **Medium Priority Tasks:** ~60 hours
- **Total Estimated:** ~140 hours (3-4 weeks full-time)

---

## Part 1: Repository Structure Analysis

### 1.1 Current Structure

```
PhD-HAR-Segmentation/
├── sashar/                      # Core implementation
│   ├── models/
│   │   ├── sas_har.py          # Main model ✅
│   │   ├── encoder.py          # CNN + Transformer ✅
│   │   ├── heads.py            # Task heads ✅
│   │   └── tcbl.py             # Self-supervised ✅
│   ├── data/
│   │   ├── base_dataset.py     # Abstract base ✅
│   │   ├── uci_har.py          # Dataset loader ✅
│   │   ├── wisdm.py            # Dataset loader ✅
│   │   ├── pamap2.py           # Dataset loader ✅
│   │   ├── opportunity.py      # Dataset loader ✅
│   │   └── [realworld.py]      # MISSING ❌
│   ├── evaluation/
│   │   └── metrics.py          # Evaluation ✅
│   ├── baselines/
│   │   └── segmentation_baselines.py ✅
│   └── utils/                   # Utilities ✅
├── configs/                     # YAML configs ⚠️ Basic
├── docs/                        # Documentation
│   ├── proposal/                # PhD proposal ✅
│   ├── theory/                  # Math framework ⚠️ Incomplete
│   ├── experiments/             # Experiment plan ✅
│   ├── literature/              # Literature review ✅
│   ├── paper_outlines.md        # Paper outlines ✅
│   └── [papers/]                # MISSING ❌
├── scripts/
│   ├── train.py                 # Training ⚠️ Basic
│   └── train_hydra.py           # Hydra training ⚠️ Incomplete
└── [tests/]                     # MISSING ❌
```

### 1.2 Proposed Enhanced Structure

```
PhD-HAR-Segmentation/
├── sashar/                      # Core package
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── sas_har.py
│   │   ├── encoder.py
│   │   ├── heads.py
│   │   ├── tcbl.py
│   │   └── distillation.py     # NEW: KD methods
│   ├── data/                    # Data handling
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── preprocessing/       # NEW: Preprocessing module
│   │   │   ├── __init__.py
│   │   │   ├── filters.py
│   │   │   ├── normalization.py
│   │   │   └── augmentation.py
│   │   ├── datasets/            # NEW: Organized datasets
│   │   │   ├── __init__.py
│   │   │   ├── uci_har.py
│   │   │   ├── wisdm.py
│   │   │   ├── pamap2.py
│   │   │   ├── opportunity.py
│   │   │   └── realworld.py    # NEW
│   │   └── datamodules.py      # NEW: PyTorch Lightning
│   ├── evaluation/              # Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── benchmark.py        # NEW: Benchmark runner
│   │   └── visualization.py    # NEW: Result plots
│   ├── baselines/               # Baseline methods
│   │   ├── __init__.py
│   │   ├── segmentation_baselines.py
│   │   └── classification_baselines.py  # NEW
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── reproducibility.py
│       ├── checkpointing.py
│       └── logging.py
├── configs/                     # Hydra configs
│   ├── default.yaml
│   ├── model/
│   │   ├── sas_har.yaml
│   │   ├── sas_har_lite.yaml
│   │   └── baselines.yaml
│   ├── dataset/
│   │   ├── uci_har.yaml
│   │   ├── wisdm.yaml
│   │   ├── pamap2.yaml
│   │   ├── opportunity.yaml
│   │   └── realworld.yaml
│   ├── experiment/
│   │   ├── paper1_tcbl.yaml
│   │   ├── paper2_sas_har.yaml
│   │   └── paper3_edge.yaml
│   └── training/
│       ├── default.yaml
│       ├── pretrain.yaml
│       └── finetune.yaml
├── scripts/                     # Executable scripts
│   ├── train.py
│   ├── evaluate.py              # NEW
│   ├── run_benchmark.py         # NEW
│   ├── download_data.py         # NEW
│   └── export_model.py          # NEW
├── tests/                       # NEW: Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_baselines.py
│   └── test_integration.py
├── docs/                        # Documentation
│   ├── proposal/
│   ├── theory/
│   │   └── mathematical_framework.md  # Enhanced
│   ├── experiments/
│   ├── literature/
│   ├── figures/                 # NEW: Publication figures
│   │   ├── architecture/
│   │   ├── pipeline/
│   │   └── results/
│   └── papers/                  # NEW: Paper manuscripts
│       ├── paper1_tcbl/
│       │   ├── main.tex
│       │   ├── figures/
│       │   └── tables/
│       ├── paper2_sas_har/
│       └── paper3_nanohar/
├── notebooks/                   # NEW: Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── .github/                     # NEW: CI/CD
│   └── workflows/
│       ├── tests.yml
│       └── lint.yml
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Part 2: Mathematical Framework Enhancement

### 2.1 Current State

The existing `mathematical_framework.md` provides basic formulations but lacks:
- Formal theorem statements
- Proof sketches
- Complexity analysis
- Convergence guarantees
- Generalization bounds

### 2.2 Proposed Enhancements

#### A. Formal Problem Definition

```latex
\begin{definition}[Temporal Segmentation Problem]
Given a multivariate time series $\mathcal{X} = \{x_1, \ldots, x_T\}$ where 
$x_t \in \mathbb{R}^C$, find a segmentation $\mathcal{S}^* = \{(s_i, e_i, y_i)\}_{i=1}^N$ 
that minimizes:

\mathcal{L}(\mathcal{S}) = \underbrace{\mathcal{L}_{seg}(\mathcal{X}, \mathcal{B})}_{\text{Segmentation Loss}} 
+ \lambda \underbrace{\mathcal{L}_{cls}(\mathcal{X}, \mathcal{S})}_{\text{Classification Loss}}
\end{definition}
```

#### B. Theorem: TCBL Convergence

```latex
\begin{theorem}[TCBL Convergence]
Under the assumptions:
\begin{enumerate}
    \item The loss function $\mathcal{L}_{TCBL}$ is $L$-Lipschitz continuous
    \item The learning rate $\eta_t$ satisfies $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$
    \item The gradient noise has bounded variance $\mathbb{E}[\|g_t - \nabla \mathcal{L}\|^2] \leq \sigma^2$
\end{enumerate}

Then SGD with learning rate $\eta_t = \frac{\eta_0}{\sqrt{t}}$ converges to a 
stationary point $\theta^*$ such that $\mathbb{E}[\|\nabla \mathcal{L}(\theta^*)\|^2] \leq \epsilon$ 
in $O(1/\epsilon^2)$ iterations.
\end{theorem}
```

#### C. Complexity Analysis Table

| Component | Time Complexity | Space Complexity | FLOPs |
|-----------|-----------------|------------------|-------|
| CNN Encoder | $O(C \cdot d \cdot K \cdot T)$ | $O(C \cdot d)$ | $C \cdot d \cdot K \cdot T$ |
| Linear Attention | $O(T \cdot d^2)$ | $O(T \cdot d)$ | $2 \cdot T \cdot d^2$ |
| TCBL Pre-training | $O(B \cdot T^2 \cdot d)$ | $O(B \cdot T \cdot d)$ | $B \cdot T^2 \cdot d$ |
| Boundary Head | $O(T \cdot d)$ | $O(d)$ | $T \cdot d$ |
| **Total (Full Model)** | $O(T \cdot d^2)$ | $O(T \cdot d)$ | $\sim 150$ MFLOPs |
| **Total (Lite Model)** | $O(T \cdot d^2)$ | $O(T \cdot d)$ | $\sim 24$ MFLOPs |

---

## Part 3: Dataset Support Enhancement

### 3.1 Missing: RealWorld HAR Dataset

**Dataset Details:**
- 15 subjects
- 8 activities
- 7 body positions
- Acc, Gyro, Mag sensors
- Publicly available

**Implementation Required:**

```python
# sashar/data/datasets/realworld.py

class RealWorldHARDataset(BaseHARDataset):
    """
    RealWorld HAR Dataset Loader
    
    URL: https://sensor.informatik.uni-mannheim.de/#dataset_realworld
    Activities: walking, running, sitting, standing, lying, climbing_up, 
                climbing_down, jumping
    Subjects: 15
    Positions: chest, forearm, head, shin, thigh, upper arm, waist
    """
    
    NUM_CLASSES = 8
    SAMPLING_RATE = 50
    SENSOR_CHANNELS = {
        'accelerometer': 3,
        'gyroscope': 3,
        'magnetometer': 3
    }
    
    ACTIVITY_LABELS = {
        0: 'walking',
        1: 'running', 
        2: 'sitting',
        3: 'standing',
        4: 'lying',
        5: 'climbing_up',
        6: 'climbing_down',
        7: 'jumping'
    }
    
    DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/datasets/RealWorldHAR"
    
    def __init__(self, root, position='waist', **kwargs):
        self.position = position
        super().__init__(root, **kwargs)
    
    def download_and_preprocess(self):
        # Implementation
        pass
```

### 3.2 Dataset Comparison Table

| Dataset | Subjects | Activities | Sensors | Hz | Duration | Support |
|---------|----------|------------|---------|-----|----------|---------|
| UCI-HAR | 30 | 6 | Acc, Gyro | 50 | 8h | ✅ |
| WISDM | 36 | 6 | Acc | 20 | 54h | ✅ |
| PAMAP2 | 9 | 18 | Acc, Gyro, Mag, HR | 100 | 10h | ✅ |
| Opportunity | 4 | 17 | 72 sensors | 30 | 6h | ✅ |
| RealWorld | 15 | 8 | Acc, Gyro, Mag | 50 | 9h | ❌ TODO |

---

## Part 4: Test Suite Implementation

### 4.1 Test Structure

```
tests/
├── conftest.py              # Fixtures and configuration
├── test_models/
│   ├── test_sas_har.py      # Model tests
│   ├── test_encoder.py
│   ├── test_heads.py
│   └── test_tcbl.py
├── test_data/
│   ├── test_datasets.py     # Dataset tests
│   ├── test_preprocessing.py
│   └── test_augmentation.py
├── test_evaluation/
│   ├── test_metrics.py      # Metrics tests
│   └── test_benchmark.py
├── test_baselines/
│   └── test_segmentation.py
└── test_integration/
    └── test_end_to_end.py   # Integration tests
```

### 4.2 Key Test Cases

```python
# tests/test_models/test_sas_har.py

import pytest
import torch
from sashar.models import SASHAR, SASHARLite

class TestSASHAR:
    """Test suite for SAS-HAR model."""
    
    @pytest.fixture
    def model(self):
        return SASHAR(input_channels=6, num_classes=6)
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 6, 128)
    
    def test_forward_shape(self, model, sample_input):
        """Test output shapes match documentation."""
        output = model(sample_input)
        
        assert output['logits'].shape == (4, 6)
        assert output['boundaries'].shape[0] == 4
        assert 'temporal_features' in output
    
    def test_boundary_threshold(self, model, sample_input):
        """Test boundary predictions with threshold."""
        boundaries = model.get_boundary_predictions(sample_input, threshold=0.5)
        assert boundaries.dtype == torch.long
        assert boundaries.shape[0] == 4
    
    def test_parameter_count(self, model):
        """Test parameter count is within expected range."""
        params = model.count_parameters()
        assert 1e5 < params < 2e6  # Between 100K and 2M
    
    def test_lite_version_smaller(self):
        """Test lite version has fewer parameters."""
        full = SASHAR(input_channels=6, num_classes=6)
        lite = SASHARLite(input_channels=6, num_classes=6)
        
        assert lite.count_parameters() < full.count_parameters()
    
    def test_gradient_flow(self, model, sample_input):
        """Test gradients flow through all parameters."""
        output = model(sample_input)
        loss = output['logits'].sum() + output['boundaries'].sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    
    def test_reproducibility(self, sample_input):
        """Test deterministic behavior with seed."""
        torch.manual_seed(42)
        model1 = SASHAR(input_channels=6, num_classes=6)
        out1 = model1(sample_input)
        
        torch.manual_seed(42)
        model2 = SASHAR(input_channels=6, num_classes=6)
        out2 = model2(sample_input)
        
        assert torch.allclose(out1['logits'], out2['logits'])
```

---

## Part 5: Publication-Quality Diagrams

### 5.1 Required Diagrams

| Diagram | Purpose | Tool | Priority |
|---------|---------|------|----------|
| SAS-HAR Architecture | Main paper figure | draw.io/TikZ | High |
| TCBL Module | Paper 1 figure | draw.io/TikZ | High |
| Segmentation Pipeline | Overview figure | draw.io | High |
| Contrastive Learning | Pre-training visualization | Matplotlib | High |
| Results Comparison | Benchmark results | Matplotlib/Seaborn | Medium |
| Attention Visualization | Interpretability | Custom | Medium |

### 5.2 Architecture Diagram Specification

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAS-HAR Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: [B, C, T]                                              │
│      │                                                           │
│      ▼                                                           │
│   ┌────────────────────────────────────────┐                    │
│   │         CNN Feature Encoder            │                    │
│   │  ┌──────────┐  ┌──────────┐  ┌──────┐ │                    │
│   │  │ DWConv1  │→│ DWConv2  │→│ GAP  │ │                    │
│   │  │ 64×5     │  │ 128×3    │  │      │ │                    │
│   │  └──────────┘  └──────────┘  └──────┘ │                    │
│   └────────────────────────────────────────┘                    │
│      │                                                           │
│      ▼ [B, T', 256]                                              │
│   ┌────────────────────────────────────────┐                    │
│   │     Linear Attention Transformer       │                    │
│   │  ┌──────────────────────────────────┐  │                    │
│   │  │  Layer 1: LinearAttn + FFN       │  │                    │
│   │  │  Layer 2: LinearAttn + FFN       │  │                    │
│   │  │  Layer 3: LinearAttn + FFN       │  │                    │
│   │  └──────────────────────────────────┘  │                    │
│   └────────────────────────────────────────┘                    │
│      │                                                           │
│      ├─────────────────────┐                                     │
│      ▼                     ▼                                     │
│   ┌────────────┐      ┌─────────────┐                          │
│   │  Semantic  │      │    TASM     │                          │
│   │  Boundary  │      │ (Trans.     │                          │
│   │  Attention │      │  Module)    │                          │
│   └────────────┘      └─────────────┘                          │
│      │                     │                                     │
│      ▼                     ▼                                     │
│   ┌────────────┐      ┌─────────────┐                          │
│   │  Boundary  │      │    Class    │                          │
│   │  Head      │      │    Head     │                          │
│   └────────────┘      └─────────────┘                          │
│      │                     │                                     │
│      ▼                     ▼                                     │
│  Boundaries [B,T',1]  Logits [B, K]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Paper Manuscript Development

### 6.1 Paper 1: TCBL (IEEE TBME)

**Title:** "Temporal Contrastive Boundary Learning: Self-Supervised Activity Segmentation for Wearable Sensors"

**Target:** IEEE Transactions on Biomedical Engineering (IF: 7.0)

**Timeline:**
- Draft completion: Month 24
- Internal review: Month 25
- Submission: Month 26
- Expected decision: Month 30

**Key Sections:**
1. Introduction (2 pages)
2. Related Work (2 pages)
3. Method - TCBL Framework (3 pages)
4. Experiments (4 pages)
5. Discussion (1 page)
6. Conclusion (0.5 pages)

**Required Figures:**
- Fig 1: Motivation diagram
- Fig 2: TCBL architecture
- Fig 3: Contrastive learning visualization
- Fig 4: Label efficiency curves
- Fig 5: Ablation study

### 6.2 Paper 2: SAS-HAR (NeurIPS)

**Title:** "SAS-HAR: Self-Supervised Attention-based Segmentation for Human Activity Recognition"

**Target:** NeurIPS 2026

**Timeline:**
- Draft completion: Month 28
- Internal review: Month 29
- Submission: Month 30 (NeurIPS deadline)

**Key Contributions:**
1. Unified segmentation-classification framework
2. Semantic Boundary Attention (SBA)
3. Transitional Activity Specialization Module (TASM)
4. State-of-the-art results on 4 benchmarks

### 6.3 Paper 3: NanoHAR (MLSys)

**Title:** "NanoHAR: Nanojoule-Level Human Activity Recognition with Self-Supervised Segmentation"

**Target:** MLSys 2027

**Key Contributions:**
1. First nanojoule-level segmentation framework
2. Knowledge distillation for HAR
3. INT8 quantization with <1% accuracy loss
4. Real-world deployment validation

---

## Part 7: Training Pipeline Enhancement

### 7.1 Hydra Configuration System

```yaml
# configs/experiment/paper1_tcbl.yaml

defaults:
  - override /model: sas_har
  - override /dataset: uci_har
  - _self_

experiment:
  name: paper1_tcbl_uci_har
  seed: 42
  tags: ["tcbl", "self-supervised", "paper1"]

model:
  hidden_dim: 256
  num_heads: 4
  num_transformer_layers: 3

training:
  pretrain_epochs: 100
  finetune_epochs: 50
  pretrain_lr: 1e-4
  finetune_lr: 1e-5
  
  # Label efficiency experiment
  label_ratios: [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

tcbl:
  temperature: 0.1
  boundary_weight: 2.0
  consistency_weight: 0.5
  
  # Pretext task weights
  contrastive_weight: 1.0
  continuity_weight: 0.5
  masked_weight: 0.3

evaluation:
  metrics: ["boundary_f1", "accuracy", "macro_f1"]
  eval_freq: 5

logging:
  wandb: true
  wandb_project: "sas-har-paper1"
  save_dir: "results/paper1/"
```

### 7.2 Experiment Runner

```python
# scripts/run_benchmark.py

import hydra
from omegaconf import DictConfig
from sashar.evaluation import BenchmarkRunner

@hydra.main(config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig):
    """Run comprehensive benchmark of all methods."""
    
    runner = BenchmarkRunner(
        datasets=cfg.datasets,
        methods=cfg.methods,
        metrics=cfg.metrics,
        output_dir=cfg.output_dir
    )
    
    # Run all experiments
    results = runner.run_all(
        n_seeds=cfg.n_seeds,
        parallel=cfg.parallel
    )
    
    # Generate tables and figures
    runner.generate_latex_tables(results)
    runner.generate_comparison_plots(results)
    
    # Save raw results
    runner.save_results(results, format="csv")

if __name__ == "__main__":
    main()
```

---

## Part 8: Implementation Checklist

### Phase 1: Foundation (Week 1-2)

- [ ] Create test suite structure
- [ ] Implement RealWorld HAR dataset loader
- [ ] Add comprehensive preprocessing module
- [ ] Set up CI/CD with GitHub Actions

### Phase 2: Core Enhancements (Week 3-4)

- [ ] Enhance mathematical framework with theorems
- [ ] Create publication-quality diagrams
- [ ] Implement benchmark runner
- [ ] Add experiment tracking (Weights & Biases)

### Phase 3: Paper Development (Week 5-8)

- [ ] Complete Paper 1 manuscript draft
- [ ] Complete Paper 2 manuscript draft
- [ ] Complete Paper 3 manuscript draft
- [ ] Generate all required figures and tables

### Phase 4: Validation (Week 9-10)

- [ ] Run all experiments with multiple seeds
- [ ] Perform statistical analysis
- [ ] Create reproducibility documentation
- [ ] Final repository cleanup

---

## Part 9: Quality Metrics

### 9.1 Code Quality Targets

| Metric | Target | Current |
|--------|--------|---------|
| Test Coverage | >80% | 0% |
| Type Hints | 100% | ~60% |
| Docstrings | 100% | ~80% |
| Linting (ruff) | 0 errors | TBD |
| Cyclomatic Complexity | <10 | TBD |

### 9.2 Documentation Targets

| Document | Status | Quality |
|----------|--------|---------|
| README | Needs update | Medium |
| API Docs | Missing | - |
| User Guide | Missing | - |
| Contributing | Missing | - |

---

## Appendix A: File Inventory

### Existing Python Files

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| sas_har.py | 312 | Main model | High |
| encoder.py | ~200 | CNN + Transformer | High |
| heads.py | ~150 | Task heads | High |
| tcbl.py | 610 | Self-supervised | High |
| base_dataset.py | 261 | Data base | High |
| metrics.py | 591 | Evaluation | High |
| segmentation_baselines.py | 510 | Baselines | High |

### Existing Documentation

| Document | Words | Purpose | Quality |
|----------|-------|---------|---------|
| phd_proposal.md | ~6000 | Research proposal | Very High |
| mathematical_framework.md | ~2000 | Theory | Medium |
| paper_outlines.md | ~4000 | Paper drafts | High |
| experiment_plan.md | ~2500 | Experiments | High |

---

## Appendix B: Dependencies

### Required Packages (requirements.txt)

```
# Core
torch>=2.0
numpy>=1.24
scipy>=1.10

# Data
pandas>=2.0
h5py>=3.8
wget>=3.2

# ML
scikit-learn>=1.2
transformers>=4.30

# Config
hydra-core>=1.3
omegaconf>=2.3

# Logging
wandb>=0.15
tensorboard>=2.13

# Visualization
matplotlib>=3.7
seaborn>=0.12

# Testing
pytest>=7.3
pytest-cov>=4.1

# Code Quality
ruff>=0.0.270
mypy>=1.3

# Deployment
onnx>=1.14
onnxruntime>=1.15
```

---

*Document Version: 1.0*  
*Last Updated: March 2026*  
*Author: AI Analysis System*
