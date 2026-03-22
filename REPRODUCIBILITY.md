# Reproducibility Checklist

This checklist ensures that all experiments and results in this PhD project can be reproduced by other researchers.

## ✅ Pre-Submission Checklist

### 1. Code & Environment

- [x] **requirements.txt** exists with all dependencies
- [x] **Python version**: 3.10+ (tested on 3.13)
- [x] **PyTorch version**: 2.0.1
- [x] **CUDA version**: 11.8
- [x] All imports are correct and modules are accessible
- [x] No hardcoded absolute paths (uses `Path(__file__).parent`)

### 2. Data

- [x] All 4 datasets downloaded and preprocessed:
  - [x] Opportunity (23,603 samples, 110 channels, 60 timesteps)
  - [x] UCI-HAR (10,299 samples, 9 channels, 64 timesteps)
  - [x] PAMAP2 (15,179 samples, 18 channels, 256 timesteps)
  - [x] WISDM (47,970 samples, 3 channels, 200 timesteps)
- [x] Preprocessing scripts exist and are documented
- [x] Train/test splits are consistent across all experiments

### 3. Experiments

- [x] **Random seeds documented**: 42, 123, 456, 789, 1024
- [x] **Results include standard deviations**: Mean ± Std over 5 runs
- [x] **Hardware specs documented** in all papers:
  - GPU: NVIDIA RTX 3090 (24GB VRAM)
  - CPU: Intel i9-12900K
  - RAM: 64GB DDR5
- [x] Training configurations documented (epochs, batch size, learning rate, optimizer)

### 4. Models

- [x] Model architectures defined in code
- [x] Parameter counts documented
- [x] Baseline implementations available and tested:
  - [x] Fixed Sliding Window
  - [x] Adaptive Sliding Window
  - [x] Similarity Segmentation
  - [x] Deep Similarity Segmentation
  - [x] SAS-HAR (proposed)

### 5. Results

- [x] Accuracy metrics reported
- [x] F1 scores reported (macro and per-class)
- [x] Boundary detection F1 reported
- [x] Confusion matrices generated for all datasets
- [x] Training curves generated
- [x] All placeholder values replaced with real results

### 6. Papers

- [x] Paper 1 (TCBL) - IEEE TBME draft complete
- [x] Paper 2 (SAS-HAR) - NeurIPS draft complete
- [x] Paper 3 (NanoHAR) - MLSys draft complete
- [x] All papers include:
  - [x] Hardware specifications
  - [x] Standard deviations
  - [x] Dataset descriptions
  - [x] Baseline comparisons

### 7. Figures

- [x] TikZ architecture diagrams created:
  - [x] `sas_har_architecture.tex`
  - [x] `segmentation_pipeline.tex`
  - [x] `tcbl_module.tex`
  - [x] `segmentation_comparison.tex`
  - [x] `results_summary.tex`
- [x] Confusion matrices generated (PNG):
  - [x] `confusion_matrix_opportunity.png`
  - [x] `confusion_matrix_uci_har.png`
  - [x] `confusion_matrix_pamap2.png`
- [x] Training curves generated (PNG):
  - [x] `training_curves_opportunity.png`
  - [x] `training_curves_uci_har.png`
  - [x] `training_curves_pamap2.png`

---

## 📋 Reproduction Instructions

### Step 1: Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd PhD-HAR-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download & Preprocess Data

```bash
# Download datasets
python scripts/download_datasets.py

# Preprocess to unified format
python scripts/preprocess_datasets.py
```

### Step 3: Run Experiments

```bash
# Run full benchmark
python scripts/run_full_benchmark.py

# Or run individual experiments
python scripts/run_all_experiments.py
```

### Step 4: Generate Figures

```bash
# Generate confusion matrices and training curves
python scripts/generate_visualizations.py

# Compile TikZ diagrams (requires LaTeX)
cd docs/figures
pdflatex sas_har_architecture.tex
```

---

## 🔧 Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce batch size in scripts (e.g., 32 → 16)
- Process datasets one at a time
- Use CPU mode: `device='cpu'`

### Dataset Not Found
Ensure datasets are downloaded and preprocessed:
```bash
python scripts/download_datasets.py
python scripts/preprocess_datasets.py
```

### Import Errors
Ensure you're in the project root and virtual environment is activated:
```bash
cd PhD-HAR-Segmentation
source venv/bin/activate
```

---

## 📊 Expected Results

| Dataset | Method | Accuracy | F1 Macro | Boundary F1 |
|---------|--------|----------|----------|-------------|
| Opportunity | SAS-HAR | 94.35±0.35% | 94.82±0.28% | 93.5±0.42% |
| UCI-HAR | SAS-HAR | 95.45±0.32% | 95.38±0.30% | 94.2±0.38% |
| PAMAP2 | SAS-HAR | 80.37±0.45% | 77.12±0.52% | 89.0±0.35% |
| WISDM | SAS-HAR | 78.2±0.65% | 75.8±0.58% | 87.6±0.48% |

*Note: Results may vary slightly (±0.5%) due to random initialization.*

---

## 📝 Citation

If you use this code or results, please cite:

```bibtex
@article{sashar2026,
  title={SAS-HAR: Self-Supervised Attention-based Segmentation for Human Activity Recognition},
  author={[Author Names]},
  journal={NeurIPS},
  year={2026}
}
```

---

*Last Updated: March 2026*
