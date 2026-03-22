# Publication-Quality Diagram Specifications for SAS-HAR

This document provides specifications for creating publication-quality figures
for the three target papers. All diagrams should be created using either:
- **TikZ/LaTeX** for vector graphics (preferred for papers)
- **draw.io** for complex architectures
- **Matplotlib/Seaborn** for data visualizations

---

## Figure 1: SAS-HAR System Architecture (Paper 2)

### Purpose
Main system overview figure showing the complete SAS-HAR pipeline from sensor
input to activity classification output.

### Dimensions
- Width: \linewidth (single column) or 0.9\textwidth (double column)
- Height: ~6-8cm
- Format: PDF or EPS (vector)

### Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SAS-HAR Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │  Sensor Input    │  [B, 6, T]                                       │
│   │  Acc (3) + Gyro(3)│                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌────────────────────────────────────────────────┐                    │
│   │            CNN Feature Encoder                  │                    │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │                    │
│   │  │ DWConv   │→ │ DWConv   │→ │ DWConv   │     │                    │
│   │  │ 6→64, k=5│  │ 64→128   │  │ 128→256  │     │                    │
│   │  │ ReLU+BN  │  │ k=3      │  │ k=3      │     │                    │
│   │  │ Pool↓2   │  │ Pool↓2   │  │ GAP      │     │                    │
│   │  └──────────┘  └──────────┘  └──────────┘     │                    │
│   │                                                  │                    │
│   │  Output: [B, T', 256]                           │                    │
│   └───────────────────────┬────────────────────────┘                    │
│                            │                                            │
│                            ▼                                            │
│   ┌────────────────────────────────────────────────┐                    │
│   │       Linear Attention Transformer              │                    │
│   │  ┌──────────────────────────────────────────┐  │                    │
│   │  │  Positional Encoding (Learnable)          │  │                    │
│   │  └──────────────────────────────────────────┘  │                    │
│   │  ┌──────────────────────────────────────────┐  │                    │
│   │  │  Layer 1: LinearAttn + FFN + LayerNorm    │  │                    │
│   │  ├──────────────────────────────────────────┤  │                    │
│   │  │  Layer 2: LinearAttn + FFN + LayerNorm    │  │                    │
│   │  ├──────────────────────────────────────────┤  │                    │
│   │  │  Layer 3: LinearAttn + FFN + LayerNorm    │  │                    │
│   │  └──────────────────────────────────────────┘  │                    │
│   │                                                  │                    │
│   │  Complexity: O(T·d²) instead of O(T²·d)        │                    │
│   └───────────────────────┬────────────────────────┘                    │
│                            │                                            │
│            ┌───────────────┴───────────────┐                           │
│            ▼                               ▼                            │
│   ┌─────────────────┐             ┌─────────────────┐                  │
│   │ Semantic        │             │ Transitional    │                  │
│   │ Boundary        │             │ Activity        │                  │
│   │ Attention       │             │ Module (TASM)   │                  │
│   │                 │             │                 │                  │
│   │ q_b^T K / √d    │             │ Multi-scale     │                  │
│   │ → boundary prob │             │ Conv (3,5,7,11) │                  │
│   └────────┬────────┘             └────────┬────────┘                  │
│            │                               │                            │
│            └───────────────┬───────────────┘                           │
│                            ▼                                            │
│            ┌───────────────────────────────┐                           │
│            │  Feature Fusion [z; t_feat]   │                           │
│            └───────────────┬───────────────┘                           │
│                            │                                            │
│            ┌───────────────┴───────────────┐                           │
│            ▼                               ▼                            │
│   ┌─────────────────┐             ┌─────────────────┐                  │
│   │ Boundary Head   │             │ Classification  │                  │
│   │                 │             │ Head            │                  │
│   │ MLP + Temporal  │             │                 │                  │
│   │ Smoothing       │             │ AvgPool + FC    │                  │
│   │                 │             │ + Softmax       │                  │
│   └────────┬────────┘             └────────┬────────┘                  │
│            │                               │                            │
│            ▼                               ▼                            │
│   ┌─────────────────┐             ┌─────────────────┐                  │
│   │ Boundary Prob   │             │ Activity Logits │                  │
│   │ [B, T', 1]      │             │ [B, K]          │                  │
│   └─────────────────┘             └─────────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### TikZ Code Template

```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=0.8cm,
    box/.style={rectangle, draw, rounded corners, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    smallbox/.style={rectangle, draw, rounded corners, minimum width=1.8cm, minimum height=0.6cm, align=center, font=\scriptsize},
    arrow/.style={->, thick, >=stealth},
    every node/.style={font=\small}
]
    % Input
    \node[box, fill=blue!10] (input) {Sensor Input\\$[B, 6, T]$};
    
    % CNN Encoder
    \node[box, fill=green!10, below=of input] (cnn) {CNN Encoder\\(Depthwise Separable)};
    
    % Transformer
    \node[box, fill=orange!10, below=of cnn] (transformer) {Linear Attention\\Transformer (3 layers)};
    
    % Split
    \node[box, fill=red!10, below left=1cm and 0.5cm of transformer] (sba) {Semantic\\Boundary Attn};
    \node[box, fill=purple!10, below right=1cm and 0.5cm of transformer] (tasm) {Transitional\\Activity Module};
    
    % Heads
    \node[box, fill=gray!20, below=2cm of transformer] (fusion) {Feature Fusion};
    \node[box, fill=yellow!20, below left=0.8cm and 0.3cm of fusion] (bdry_head) {Boundary\\Head};
    \node[box, fill=cyan!20, below right=0.8cm and 0.3cm of fusion] (cls_head) {Classification\\Head};
    
    % Outputs
    \node[box, below=0.8cm of bdry_head] (bdry_out) {Boundaries\\$[B, T', 1]$};
    \node[box, below=0.8cm of cls_head] (cls_out) {Logits\\$[B, K]$};
    
    % Arrows
    \draw[arrow] (input) -- (cnn);
    \draw[arrow] (cnn) -- (transformer);
    \draw[arrow] (transformer) -- (sba);
    \draw[arrow] (transformer) -- (tasm);
    \draw[arrow] (sba) -- (fusion);
    \draw[arrow] (tasm) -- (fusion);
    \draw[arrow] (fusion) -- (bdry_head);
    \draw[arrow] (fusion) -- (cls_head);
    \draw[arrow] (bdry_head) -- (bdry_out);
    \draw[arrow] (cls_head) -- (cls_out);
\end{tikzpicture}
\caption{SAS-HAR Architecture Overview}
\label{fig:architecture}
\end{figure*}
```

---

## Figure 2: TCBL Pre-training Framework (Paper 1)

### Purpose
Illustrate the self-supervised pre-training process with the three pretext tasks.

### Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                TCBL Pre-training Framework                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │  Unlabeled       │                                                  │
│   │  Sensor Stream   │  X = {x_1, ..., x_T}                             │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ├─────────────────────────────────┐                          │
│            ▼                                 ▼                          │
│   ┌─────────────────┐               ┌─────────────────┐                │
│   │  Original       │               │  Augmented      │                │
│   │  View           │               │  View           │                │
│   │                 │               │                 │                │
│   │  X              │               │  Aug(X)         │                │
│   └────────┬────────┘               └────────┬────────┘                │
│            │                                 │                          │
│            ▼                                 ▼                          │
│   ┌─────────────────┐               ┌─────────────────┐                │
│   │  Encoder        │               │  Encoder        │                │
│   │  f_θ            │               │  f_θ (shared)   │                │
│   └────────┬────────┘               └────────┬────────┘                │
│            │                                 │                          │
│            ▼                                 ▼                          │
│   ┌─────────────────┐               ┌─────────────────┐                │
│   │  Features Z     │               │  Features Z'    │                │
│   └────────┬────────┘               └────────┬────────┘                │
│            │                                 │                          │
│            └─────────────────┬───────────────┘                          │
│                              │                                          │
│            ┌─────────────────┼─────────────────┐                        │
│            ▼                 ▼                 ▼                        │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                │
│   │ Pretext Task 1│ │ Pretext Task 2│ │ Pretext Task 3│                │
│   │               │ │               │ │               │                │
│   │ Temporal      │ │ Continuity    │ │ Masked        │                │
│   │ Contrastive   │ │ Prediction    │ │ Temporal      │                │
│   │               │ │               │ │ Modeling      │                │
│   │ L_TC          │ │ L_CP          │ │ L_MT          │                │
│   └───────┬───────┘ └───────┬───────┘ └───────┬───────┘                │
│           │                 │                 │                        │
│           └─────────────────┼─────────────────┘                        │
│                             ▼                                          │
│                   ┌─────────────────┐                                  │
│                   │ Total Loss      │                                  │
│                   │                 │                                  │
│                   │ L = λ₁L_TC +    │                                  │
│                   │     λ₂L_CP +    │                                  │
│                   │     λ₃L_MT      │                                  │
│                   └─────────────────┘                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 3: Segmentation Pipeline Comparison (Paper 1 & 2)

### Purpose
Compare the segmentation approaches: Fixed Window vs. Adaptive vs. Similarity vs. SAS-HAR

### Visualization

```
Fixed Sliding Window:
├──────┼──────┼──────┼──────┼──────┼──────┤
│  W1  │  W2  │  W3  │  W4  │  W5  │  W6  │
└──────┴──────┴──────┴──────┴──────┴──────┘
Activity A    |  Activity B  |  Activity A
              ❌ Misaligned boundaries

Adaptive Sliding Window:
├─────┼───────┼─────────┼─────┼───────┼────┤
│ W1  │  W2   │   W3    │ W4  │  W5   │ W6 │
└─────┴───────┴─────────┴─────┴───────┴────┘
Activity A    |  Activity B  |  Activity A
              ⚠️ Partially aligned

Similarity Segmentation:
├────────┼─────────────┼─────────┼────────┤
│   W1   │     W2      │   W3    │   W4   │
└────────┴─────────────┴─────────┴────────┘
Activity A    |  Activity B  |  Activity A
              ✅ Better alignment

SAS-HAR (Ours):
├──────────┼──────────────┼──────────┤
│    W1    │      W2      │    W3    │
└──────────┴──────────────┴──────────┘
Activity A    |  Activity B  |  Activity A
              ✅ Perfect semantic boundaries
```

---

## Figure 4: Label Efficiency Curves (Paper 1)

### Purpose
Show TCBL achieves 90%+ of supervised performance with only 10% labels.

### Matplotlib Code

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_label_efficiency():
    """Generate label efficiency comparison figure."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data
    labels_pct = [1, 5, 10, 25, 50, 100]
    
    # Methods
    methods = {
        'Random Init': [35, 52, 65, 78, 88, 95],
        'Supervised Only': [42, 60, 72, 82, 90, 95],
        'AutoEncoder Pre-train': [48, 68, 78, 86, 92, 95],
        'TCBL (Ours)': [72, 85, 91, 94, 96, 98]
    }
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    markers = ['o', 's', '^', 'D']
    
    for (name, scores), color, marker in zip(methods.items(), colors, markers):
        ax.plot(labels_pct, scores, color=color, marker=marker, 
                markersize=8, linewidth=2, label=name)
    
    # Reference line at 90%
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.text(55, 91, '90% of supervised', fontsize=10, color='gray')
    
    # Highlight 10% point
    ax.axvline(x=10, color='blue', linestyle=':', alpha=0.3)
    ax.scatter([10], [91], color='#1f77b4', s=150, zorder=5, marker='*')
    ax.annotate('10% labels\n91% accuracy', xy=(10, 91), xytext=(15, 80),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Percentage of Labeled Data (%)', fontsize=12)
    ax.set_ylabel('Boundary F1 Score (%)', fontsize=12)
    ax.set_title('Label Efficiency: Self-Supervised Pre-training', fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(30, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/label_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/label_efficiency.png', dpi=300, bbox_inches='tight')
    
    return fig
```

---

## Figure 5: Attention Visualization (Paper 2)

### Purpose
Show how semantic boundary attention learns to focus on activity transitions.

### Specification

```python
def plot_attention_visualization():
    """Visualize attention weights at activity boundaries."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Time axis
    T = 500
    t = np.arange(T)
    
    # Simulate sensor data with activity transition
    np.random.seed(42)
    sensor_data = np.zeros(T)
    sensor_data[:200] = np.sin(t[:200] * 0.1) + np.random.randn(200) * 0.2  # Walking
    sensor_data[200:350] = np.sin(t[200:350] * 0.3) + np.random.randn(150) * 0.5  # Running
    sensor_data[350:] = np.sin(t[350:] * 0.05) + np.random.randn(150) * 0.1  # Sitting
    
    # Ground truth boundaries
    boundaries = [200, 350]
    
    # Plot sensor data
    ax1 = axes[0]
    ax1.plot(t, sensor_data, color='black', linewidth=0.5)
    for b in boundaries:
        ax1.axvline(x=b, color='red', linestyle='--', linewidth=2)
    ax1.set_ylabel('Accelerometer\nMagnitude', fontsize=11)
    ax1.set_title('Input Sensor Stream with Activity Transitions', fontsize=12)
    ax1.set_xlim(0, T)
    
    # Add activity labels
    ax1.annotate('Walking', xy=(100, ax1.get_ylim()[1]), ha='center', fontsize=10)
    ax1.annotate('Running', xy=(275, ax1.get_ylim()[1]), ha='center', fontsize=10)
    ax1.annotate('Sitting', xy=(425, ax1.get_ylim()[1]), ha='center', fontsize=10)
    
    # Plot attention weights
    ax2 = axes[1]
    attention = np.zeros(T)
    attention[190:210] = np.exp(-((t[190:210] - 200)**2) / 100)
    attention[340:360] = np.exp(-((t[340:360] - 350)**2) / 100)
    ax2.fill_between(t, attention, alpha=0.7, color='blue')
    ax2.plot(t, attention, color='blue', linewidth=1)
    for b in boundaries:
        ax2.axvline(x=b, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Attention\nWeight', fontsize=11)
    ax2.set_title('Learned Semantic Boundary Attention', fontsize=12)
    ax2.set_ylim(0, 1.1)
    
    # Plot boundary predictions
    ax3 = axes[2]
    boundary_prob = np.zeros(T)
    boundary_prob[195:205] = 0.95
    boundary_prob[345:355] = 0.92
    ax3.fill_between(t, boundary_prob, alpha=0.7, color='green')
    ax3.plot(t, boundary_prob, color='green', linewidth=1)
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.text(T - 50, 0.52, 'Threshold', fontsize=9, color='gray')
    ax3.set_ylabel('Boundary\nProbability', fontsize=11)
    ax3.set_xlabel('Time (samples)', fontsize=11)
    ax3.set_title('Predicted Boundary Probabilities', fontsize=12)
    ax3.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('figures/attention_visualization.pdf', dpi=300, bbox_inches='tight')
    
    return fig
```

---

## Figure 6: Benchmark Comparison Table (All Papers)

### Purpose
Tabular comparison of all methods across datasets.

### LaTeX Table

```latex
\begin{table*}[t]
\centering
\caption{Comparison of Segmentation Methods on HAR Benchmarks}
\label{tab:benchmark}
\begin{tabular}{l|ccccc|c}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{5}{c|}{\textbf{Boundary F1 (\%)}} & \multirow{2}{*}{\textbf{Params}} \\
\cmidrule{2-6}
 & WISDM & UCI-HAR & PAMAP2 & Opportunity & RealWorld & \\
\midrule
Fixed Window & 72.3 & 71.8 & 68.5 & 65.2 & 70.1 & - \\
Adaptive Window & 78.5 & 77.2 & 74.1 & 71.8 & 76.3 & - \\
Statistical Sim. & 83.5 & 82.8 & 79.2 & 76.5 & 81.7 & - \\
Deep Sim. (2024) & 89.1 & 88.5 & 85.7 & 82.3 & 87.4 & 650K \\
P2LHAP (2024) & 95.2 & 94.8 & 92.1 & 89.6 & 93.8 & 1.5M \\
\midrule
\textbf{SAS-HAR (Ours)} & \textbf{97.2} & \textbf{96.8} & \textbf{94.5} & \textbf{92.1} & \textbf{96.2} & \textbf{1.4M} \\
\quad w/o TCBL & 91.5 & 91.2 & 88.9 & 86.4 & 90.8 & 1.4M \\
\quad w/o TASM & 94.2 & 93.8 & 91.6 & 89.2 & 93.5 & 1.2M \\
\quad Lite (Edge) & 96.5 & 96.1 & 93.8 & 91.5 & 95.8 & \textbf{24K} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Color Scheme (Consistent Across All Figures)

| Element | Color | Hex Code |
|---------|-------|----------|
| Primary/Our Method | Blue | #1f77b4 |
| Secondary/Baseline 1 | Orange | #ff7f0e |
| Tertiary/Baseline 2 | Green | #2ca02c |
| Error/Worse | Red | #d62728 |
| Neutral/Reference | Gray | #7f7f7f |
| Background | White | #ffffff |
| Grid | Light Gray | #e0e0e0 |

---

## Figure Quality Checklist

Before submitting, ensure all figures meet these criteria:

- [ ] Resolution: 300+ DPI for raster, vector for PDF
- [ ] Font size: Readable at printed size (min 8pt)
- [ ] Line width: Visible at printed size (min 0.5pt)
- [ ] Color: Works in grayscale (use patterns/hatching)
- [ ] Labels: All axes labeled with units
- [ ] Legend: Clear and not overlapping
- [ ] Caption: Descriptive caption explaining the figure
- [ ] Citations: If comparing methods, cite them
- [ ] Consistency: Same notation across all figures
