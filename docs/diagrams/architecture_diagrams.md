# SAS-HAR Architecture Diagrams

This document provides detailed architecture diagrams for the SAS-HAR (Self-Supervised Attention-based Segmentation for Human Activity Recognition) framework using Mermaid syntax.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [SAS-HAR Model Architecture](#2-sas-har-model-architecture)
3. [CNN Feature Encoder](#3-cnn-feature-encoder)
4. [Efficient Linear Attention Transformer](#4-efficient-linear-attention-transformer)
5. [Task-Specific Heads](#5-task-specific-heads)
6. [TCBL Self-Supervised Module](#6-tcbl-self-supervised-module)
7. [Training Pipeline](#7-training-pipeline)
8. [Edge Deployment Pipeline](#8-edge-deployment-pipeline)

---

## 1. System Overview

The complete SAS-HAR pipeline from raw sensor data to activity predictions.

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        S1[Accelerometer X,Y,Z]
        S2[Gyroscope X,Y,Z]
        S3[Optional: Magnetometer]
    end
    
    subgraph Preprocessing["⚙️ Preprocessing"]
        N1[Normalization]
        N2[Noise Filtering<br/>Butterworth 20Hz]
        N3[Windowing<br/>Sliding/Adaptive]
    end
    
    subgraph SAS_HAR["🧠 SAS-HAR Model"]
        CNN[CNN Feature Encoder]
        TRANS[Transformer Temporal Module]
        HEADS[Task-Specific Heads]
    end
    
    subgraph Outputs["📤 Outputs"]
        B1[Boundary Probabilities]
        C1[Activity Classification]
        S1[Segment Boundaries]
    end
    
    S1 --> N1
    S2 --> N1
    S3 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> CNN
    CNN --> TRANS
    TRANS --> HEADS
    HEADS --> B1
    HEADS --> C1
    B1 --> S1
    C1 --> S1
    
    style SAS_HAR fill:#e1f5fe
    style Input fill:#fff3e0
    style Outputs fill:#e8f5e9
```

---

## 2. SAS-HAR Model Architecture

The core model architecture combining CNN encoder, Transformer temporal module, and task heads.

```mermaid
flowchart TB
    subgraph Input["Input Tensor"]
        X["X ∈ ℝ<sup>B×C×T</sup><br/>B: Batch, C: Channels, T: Time"]
    end
    
    subgraph CNNEncoder["CNN Feature Encoder"]
        direction TB
        B1["Block 1<br/>DepthwiseSep Conv 6→64<br/>BN + GELU + Dropout<br/>MaxPool T→T/2"]
        B2["Block 2<br/>DepthwiseSep Conv 64→128<br/>BN + GELU + Dropout<br/>MaxPool T/2→T/4"]
        B3["Block 3<br/>DepthwiseSep Conv 128→256<br/>BN + GELU + Dropout<br/>MaxPool T/4→T/8"]
    end
    
    subgraph Transpose["Transpose"]
        TP["ℝ<sup>B×256×T/8</sup> → ℝ<sup>B×T/8×256</sup>"]
    end
    
    subgraph Transformer["Transformer Temporal Module"]
        PE["Positional Encoding<br/>Learnable ℝ<sup>1×max_seq×256</sup>"]
        
        subgraph Layers["Transformer Blocks × 3"]
            direction TB
            LN1["LayerNorm"]
            ATTN["Efficient Linear Attention<br/>O(n) complexity"]
            ADD1["Add (Residual)"]
            LN2["LayerNorm"]
            MLP["MLP<br/>256 → 512 → 256"]
            ADD2["Add (Residual)"]
        end
        
        LN3["LayerNorm"]
    end
    
    subgraph Heads["Task-Specific Heads"]
        direction LR
        subgraph BoundaryHead["Boundary Head"]
            BA["Boundary Attention<br/>Multi-Head Attention"]
            BM["MLP Layers<br/>256→128→64→1"]
            TC["Temporal Conv<br/>1→16→1"]
            SIG["Sigmoid"]
        end
        
        subgraph TransModule["Transitional Activity Module<br/>(Optional)"]
            TE["Transition Encoder<br/>BiLSTM 64 units"]
        end
        
        subgraph ClassHead["Classification Head"]
            direction TB
            POOL["Attention Pooling<br/>256→64→1"]
            CLS["MLP Classifier<br/>256→128→6"]
        end
    end
    
    subgraph Outputs["Outputs"]
        direction LR
        OUT1["Boundaries<br/>ℝ<sup>B×T/8</sup>"]
        OUT2["Logits<br/>ℝ<sup>B×6</sup>"]
    end
    
    X --> B1
    B1 --> B2
    B2 --> B3
    B3 --> TP
    TP --> PE
    PE --> LN1
    LN1 --> ATTN
    ATTN --> ADD1
    ADD1 --> LN2
    LN2 --> MLP
    MLP --> ADD2
    ADD2 --> LN3
    LN3 --> BA
    BA --> BM
    BM --> TC
    TC --> SIG
    SIG --> OUT1
    
    LN3 --> TE
    TE --> CLS
    LN3 --> POOL
    POOL --> CLS
    CLS --> OUT2
    
    style CNNEncoder fill:#bbdefb
    style Transformer fill:#c8e6c9
    style Heads fill:#ffe0b2
    style Input fill:#f3e5f5
    style Outputs fill:#e1f5fe
```

---

## 3. CNN Feature Encoder

Detailed view of the depthwise separable convolution blocks.

```mermaid
flowchart LR
    subgraph Input["Input"]
        IN["Sensor Data<br/>ℝ<sup>B×6×T</sup>"]
    end
    
    subgraph Block1["Block 1: Local Features"]
        direction TB
        D1["Depthwise Conv<br/>k=5, groups=6"]
        P1["Pointwise Conv<br/>6→64"]
        BN1["BatchNorm1d"]
        G1["GELU"]
        DO1["Dropout 0.1"]
        MP1["MaxPool1d<br/>k=2, s=2"]
        OUT1["Output: ℝ<sup>B×64×T/2</sup>"]
    end
    
    subgraph Block2["Block 2: Mid Features"]
        direction TB
        D2["Depthwise Conv<br/>k=3, groups=64"]
        P2["Pointwise Conv<br/>64→128"]
        BN2["BatchNorm1d"]
        G2["GELU"]
        DO2["Dropout 0.1"]
        MP2["MaxPool1d<br/>k=2, s=2"]
        OUT2["Output: ℝ<sup>B×128×T/4</sup>"]
    end
    
    subgraph Block3["Block 3: High-Level Features"]
        direction TB
        D3["Depthwise Conv<br/>k=3, groups=128"]
        P3["Pointwise Conv<br/>128→256"]
        BN3["BatchNorm1d"]
        G3["GELU"]
        DO3["Dropout 0.1"]
        MP3["MaxPool1d<br/>k=2, s=2"]
        OUT3["Output: ℝ<sup>B×256×T/8</sup>"]
    end
    
    IN --> D1 --> P1 --> BN1 --> G1 --> DO1 --> MP1 --> OUT1
    OUT1 --> D2 --> P2 --> BN2 --> G2 --> DO2 --> MP2 --> OUT2
    OUT2 --> D3 --> P3 --> BN3 --> G3 --> DO3 --> MP3 --> OUT3
    
    style Block1 fill:#e3f2fd
    style Block2 fill:#e8f5e9
    style Block3 fill:#fff3e0
```

### Depthwise Separable Convolution Detail

```mermaid
flowchart LR
    subgraph Standard["Standard Convolution"]
        SC["Conv1d<br/>C<sub>in</sub>→C<sub>out</sub><br/>Params: C<sub>in</sub>×C<sub>out</sub>×k"]
    end
    
    subgraph DepthwiseSeparable["Depthwise Separable Convolution"]
        DW["Depthwise Conv<br/>Groups=C<sub>in</sub><br/>Params: C<sub>in</sub>×k"]
        PW["Pointwise Conv<br/>1×1 Conv<br/>Params: C<sub>in</sub>×C<sub>out</sub>"]
        TOTAL["Total Params:<br/>C<sub>in</sub>×k + C<sub>in</sub>×C<sub>out</sub><br/>≈ 8-10× reduction"]
    end
    
    SC -.-> |"vs"| DepthwiseSeparable
    DW --> PW --> TOTAL
    
    style DepthwiseSeparable fill:#c8e6c9
    style Standard fill:#ffcdd2
```

---

## 4. Efficient Linear Attention Transformer

The O(n) complexity attention mechanism enabling long sequence processing.

```mermaid
flowchart TB
    subgraph Input["Input Features"]
        X["X ∈ ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph Projections["Linear Projections"]
        Q["Q = W<sub>q</sub>X<br/>ℝ<sup>B×T×D</sup>"]
        K["K = W<sub>k</sub>X<br/>ℝ<sup>B×T×D</sup>"]
        V["V = W<sub>v</sub>X<br/>ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph Reshape["Multi-Head Reshape"]
        QH["Q: ℝ<sup>B×H×T×d</sub><sub>h</sub>""]
        KH["K: ℝ<sup>B×H×T×d</sub><sub>h</sub>""]
        VH["V: ℝ<sup>B×H×T×d</sub><sub>h</sub>""]
    end
    
    subgraph Kernel["Kernel Function"]
        KF["φ(x) = elu(x) + 1<br/>Ensures positive values"]
    end
    
    subgraph LinearAttn["Linear Attention Computation"]
        direction TB
        KV["K<sup>T</sup>V<br/>ℝ<sup>B×H×d<sub>h</sub>×d<sub>h</sub></sup><br/>O(d²)"]
        QKV["Q(K<sup>T</sup>V)<br/>ℝ<sup>B×H×T×d<sub>h</sub></sup><br/>O(Td)"]
        NORM["Normalize<br/>Q·K<sup>T</sup>·1"]
    end
    
    subgraph Output["Output"]
        RES["Reshape: ℝ<sup>B×T×D</sup>"]
        PROJ["Output Projection<br/>W<sub>o</sub>"]
        OUT["Output: ℝ<sup>B×T×D</sup>"]
    end
    
    X --> Q
    X --> K
    X --> V
    Q --> QH
    K --> KH
    V --> VH
    QH --> KF
    KH --> KF
    KF --> KV
    VH --> KV
    KV --> QKV
    QKV --> NORM
    NORM --> RES
    RES --> PROJ
    PROJ --> OUT
    
    style LinearAttn fill:#e8f5e9
    style Kernel fill:#fff3e0
```

### Complexity Comparison

```mermaid
flowchart LR
    subgraph Standard["Standard Attention"]
        SA["Softmax(QK<sup>T</sup>/√d)V<br/>Time: O(T²d)<br/>Space: O(T²)"]
    end
    
    subgraph Linear["Linear Attention"]
        LA["Q(K<sup>T</sup>V)/normalizer<br/>Time: O(Td²)<br/>Space: O(Td + d²)"]
    end
    
    Standard --> |"T=1000: 1M ops"| SA
    Linear --> |"T=1000: 256K ops"| LA
    
    SA -.-> |"4× faster for T=1000"| LA
    
    style Linear fill:#c8e6c9
    style Standard fill:#ffcdd2
```

---

## 5. Task-Specific Heads

### Boundary Detection Head

```mermaid
flowchart TB
    subgraph Input["Temporal Features"]
        F["F ∈ ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph BoundaryAttention["Boundary Query Attention"]
        Q["Learnable Query<br/>ℝ<sup>1×1×D</sup>"]
        MHA["Multi-Head Attention<br/>4 heads"]
        ATTN["Attention Weights<br/>ℝ<sup>B×1×T</sup>"]
        REFINE["Feature Refinement<br/>F + 0.1 × Attention(F)"]
    end
    
    subgraph MLP["MLP Layers"]
        L1["Linear D→256<br/>LayerNorm + GELU + Dropout"]
        L2["Linear 256→128<br/>LayerNorm + GELU + Dropout"]
        L3["Linear 128→1"]
    end
    
    subgraph TemporalSmoothing["Temporal Smoothing"]
        TC1["Conv1d 1→16, k=5<br/>GELU"]
        TC2["Conv1d 16→1, k=5"]
    end
    
    subgraph Output["Output"]
        SIG["Sigmoid"]
        OUT["Boundary Probabilities<br/>p ∈ [0,1]<sup>B×T</sup>"]
    end
    
    F --> Q
    Q --> MHA
    F --> MHA
    MHA --> ATTN
    ATTN --> REFINE
    REFINE --> L1
    L1 --> L2
    L2 --> L3
    L3 --> TC1
    TC1 --> TC2
    TC2 --> SIG
    SIG --> OUT
    
    style BoundaryAttention fill:#e3f2fd
    style TemporalSmoothing fill:#fff3e0
```

### Classification Head

```mermaid
flowchart TB
    subgraph Input["Temporal Features"]
        F["F ∈ ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph Pooling["Attention Pooling"]
        direction TB
        A1["Linear D→D/4"]
        TANH["Tanh"]
        A2["Linear D/4→1"]
        SOFT["Softmax over T"]
        WEIGHT["Weighted Sum<br/>ℝ<sup>B×D</sup>"]
    end
    
    subgraph Classifier["MLP Classifier"]
        C1["Linear D→128<br/>LayerNorm + GELU + Dropout 0.2"]
        C2["Linear 128→C<br/>C = num_classes"]
    end
    
    subgraph Output["Output"]
        LOGITS["Logits<br/>ℝ<sup>B×C</sup>"]
        PRED["Predictions<br/>argmax over C"]
    end
    
    F --> A1 --> TANH --> A2 --> SOFT --> WEIGHT
    F --> WEIGHT
    WEIGHT --> C1 --> C2 --> LOGITS --> PRED
    
    style Pooling fill:#e8f5e9
    style Classifier fill:#e3f2fd
```

---

## 6. TCBL Self-Supervised Module

The Temporal Contrastive Boundary Learning module for self-supervised pre-training.

```mermaid
flowchart TB
    subgraph Input["Original Features"]
        F["F ∈ ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph Augmentation["Data Augmentation"]
        direction LR
        J["Jitter<br/>Gaussian Noise"]
        S["Scaling<br/>×[0.9, 1.1]"]
        TW["Time Warp<br/>Smooth Warping"]
        R["Rotation<br/>3D Rotation"]
        CD["Channel Dropout<br/>10% channels"]
        AUG["Augmented Features<br/>F' ∈ ℝ<sup>B×T×D</sup>"]
    end
    
    subgraph Projection["Projection Head"]
        P1["Linear D→D"]
        BN["BatchNorm1d"]
        RE["ReLU"]
        P2["Linear D→128"]
        Z1["z₁ ∈ ℝ<sup>B×T×128</sup>"]
        Z2["z₂ ∈ ℝ<sup>B×T×128</sup>"]
    end
    
    subgraph PseudoLabels["Pseudo Label Generation"]
        SIM["Feature Similarity<br/>cos(F<sub>t</sub>, F<sub>t+1</sub>)"]
        THRESH["Threshold<br/>1 - sim > τ"]
        BP["Boundary Pseudo<br/>b̂ ∈ {0,1}<sup>B×T</sup>"]
        
        CLUS["Random Centroids"]
        ASSIGN["Cluster Assignment"]
        AP["Activity Pseudo<br/>â ∈ {0,...,K-1}<sup>B×T</sup>"]
    end
    
    subgraph Losses["Contrastive Losses"]
        direction TB
        
        subgraph Temporal["Temporal Contrastive"]
            TC["InfoNCE Loss<br/>Positive: same activity + temporal neighbors<br/>Negative: different activities"]
        end
        
        subgraph Boundary["Boundary Contrastive"]
            BC["Margin Loss<br/>Maximize intra-class sim<br/>Minimize inter-class sim"]
        end
        
        subgraph Consistency["Temporal Consistency"]
            CONS["Smoothness + Sharpness<br/>Smooth within segments<br/>Sharp at boundaries"]
        end
    end
    
    subgraph TotalLoss["Total Loss"]
        TL["L = L<sub>TC</sub> + L<sub>BC</sub> + λ·L<sub>CONS</sub>"]
    end
    
    F --> J --> S --> TW --> R --> CD --> AUG
    F --> P1 --> BN --> RE --> P2 --> Z1
    AUG --> P1
    AUG --> BN
    Z1 --> TC
    F --> SIM --> THRESH --> BP
    BP --> TC
    F --> CLUS --> ASSIGN --> AP
    AP --> TC
    Z1 --> BC
    BP --> BC
    F --> CONS
    BP --> CONS
    AP --> CONS
    TC --> TL
    BC --> TL
    CONS --> TL
    
    style Augmentation fill:#e8f5e9
    style Losses fill:#fff3e0
    style PseudoLabels fill:#e3f2fd
```

### Augmentation Strategies

```mermaid
flowchart LR
    subgraph Original["Original Signal"]
        O["📊 Sensor Data"]
    end
    
    subgraph Augs["Augmentations"]
        J["🔹 Jitter<br/>Add ε ~ N(0, σ²)<br/>σ = 0.01"]
        SC["🔹 Scaling<br/>×α, α ∈ [0.9, 1.1]"]
        TW["🔹 Time Warp<br/>Smooth deformation"]
        R["🔹 Rotation<br/>Random 3D rotation"]
        D["🔹 Dropout<br/>Drop 10% channels"]
    end
    
    Original --> J
    Original --> SC
    Original --> TW
    Original --> R
    Original --> D
    
    J --> |"50%"| AUG["Augmented"]
    SC --> |"50%"| AUG
    TW --> |"30%"| AUG
    R --> |"50%"| AUG
    D --> |"20%"| AUG
    
    style Original fill:#e3f2fd
    style Augs fill:#fff3e0
```

---

## 7. Training Pipeline

Complete training workflow from data loading to model evaluation.

```mermaid
flowchart TB
    subgraph Data["Data Preparation"]
        direction LR
        D1["Dataset Loading<br/>UCI-HAR/WISDM/PAMAP2"]
        D2["Preprocessing<br/>Normalize + Filter"]
        D3["Windowing<br/>Sliding/Adaptive"]
        D4["DataLoader<br/>Batch + Shuffle"]
    end
    
    subgraph Pretrain["Optional: TCBL Pre-training"]
        P1["Load Unlabeled Data"]
        P2["Generate Augmentations"]
        P3["Compute Contrastive Loss"]
        P4["Update Encoder Weights"]
    end
    
    subgraph Training["Supervised Training"]
        direction TB
        T1["Forward Pass"]
        T2["Compute Losses<br/>L<sub>cls</sub> + λ·L<sub>bnd</sub>"]
        T3["Backward Pass"]
        T4["Gradient Clipping<br/>max_norm = 1.0"]
        T5["Optimizer Step<br/>AdamW"]
        T6["Scheduler Step<br/>CosineAnnealing"]
    end
    
    subgraph Validation["Validation"]
        V1["Evaluate on Val Set"]
        V2["Compute Metrics<br/>Acc, F1, Boundary F1"]
        V3["Early Stopping<br/>patience = 10"]
        V4["Save Best Model"]
    end
    
    subgraph Eval["Final Evaluation"]
        E1["Load Best Model"]
        E2["Test Set Evaluation"]
        E3["LOSO Cross-Validation"]
        E4["Generate Report"]
    end
    
    D1 --> D2 --> D3 --> D4
    D4 --> P1
    P1 --> P2 --> P3 --> P4
    P4 --> T1
    D4 --> T1
    T1 --> T2 --> T3 --> T4 --> T5 --> T6
    T6 --> V1
    V1 --> V2 --> V3
    V3 --> |"Continue"| T1
    V3 --> |"Stop"| V4
    V4 --> E1 --> E2 --> E3 --> E4
    
    style Data fill:#e3f2fd
    style Training fill:#e8f5e9
    style Validation fill:#fff3e0
    style Eval fill:#fce4ec
```

---

## 8. Edge Deployment Pipeline

Model optimization pipeline for edge device deployment.

```mermaid
flowchart TB
    subgraph Teacher["Teacher Model"]
        T1["SAS-HAR Full<br/>~1.4M parameters<br/>256 hidden dim"]
    end
    
    subgraph Distillation["Knowledge Distillation"]
        direction TB
        D1["Create Student<br/>SASHARLite<br/>~150K parameters"]
        D2["Forward Pass<br/>Both Models"]
        D3["Compute Distillation Loss<br/>L<sub>KD</sub> = KL(soft_teacher || soft_student)"]
        D4["Train Student<br/>α·L<sub>hard</sub> + (1-α)·L<sub>soft</sub>"]
    end
    
    subgraph Quantization["INT8 Quantization"]
        direction TB
        Q1["Calibration<br/>Run on representative data"]
        Q2["Quantize Weights<br/>float32 → int8"]
        Q3["Quantize Activations<br/>Dynamic/Static"]
        Q4["Fuse Operations<br/>Conv+BN+ReLU"]
    end
    
    subgraph Optimization["Model Optimization"]
        O1["ONNX Export"]
        O2["TensorRT Optimization<br/>(NVIDIA)"]
        O3["TFLite Conversion<br/>(ARM Cortex-M)"]
        O4["Pruning<br/>Remove 20% weights"]
    end
    
    subgraph Deployment["Edge Deployment"]
        direction LR
        E1["NVIDIA Jetson<br/>TensorRT<br/>~5ms latency"]
        E2["Raspberry Pi 4<br/>ONNX Runtime<br/>~15ms latency"]
        E3["ARM Cortex-M4<br/>TFLite Micro<br/>~50ms latency"]
    end
    
    subgraph Metrics["Deployment Metrics"]
        M1["Model Size<br/>< 500KB"]
        M2["Latency<br/>< 50ms"]
        M3["Energy<br/>< 45 nJ/sample"]
        M4["Accuracy Drop<br/>< 2%"]
    end
    
    T1 --> D1
    D1 --> D2
    T1 --> D2
    D2 --> D3 --> D4
    D4 --> Q1 --> Q2 --> Q3 --> Q4
    Q4 --> O1
    O1 --> O2
    O1 --> O3
    O1 --> O4
    O2 --> E1
    O3 --> E3
    O4 --> E2
    E1 --> M1
    E1 --> M2
    E2 --> M3
    E3 --> M4
    
    style Teacher fill:#e3f2fd
    style Distillation fill:#e8f5e9
    style Quantization fill:#fff3e0
    style Deployment fill:#fce4ec
```

### Model Size Comparison

```mermaid
flowchart LR
    subgraph Models["Model Variants"]
        M1["SAS-HAR Full<br/>1.4M params<br/>5.6 MB<br/>Acc: 96.2%"]
        M2["SAS-HAR Lite<br/>150K params<br/>600 KB<br/>Acc: 94.8%"]
        M3["SAS-HAR Quantized<br/>150K params<br/>150 KB<br/>Acc: 94.1%"]
        M4["SAS-HAR Pruned<br/>120K params<br/>120 KB<br/>Acc: 93.5%"]
    end
    
    M1 --> |"Distillation"| M2
    M2 --> |"INT8 Quant"| M3
    M3 --> |"Pruning 20%"| M4
    
    style M1 fill:#ffcdd2
    style M2 fill:#c8e6c9
    style M3 fill:#bbdefb
    style M4 fill:#e1bee7
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 📥 | Input layer |
| ⚙️ | Processing/Transformation |
| 🧠 | Neural network component |
| 📤 | Output layer |
| 🔹 | Augmentation operation |
| 📊 | Data/Signal |

## Dimension Notation

- **B**: Batch size
- **C**: Number of channels (typically 6 for acc+gyro)
- **T**: Time steps (sequence length)
- **D**: Feature dimension (hidden size)
- **H**: Number of attention heads
- **d<sub>h</sub>**: Head dimension (D/H)
- **K**: Number of activity classes

## File References

| Component | File Path |
|-----------|-----------|
| SAS-HAR Model | `sashar/models/sas_har.py` |
| CNN Encoder | `sashar/models/encoder.py` |
| Task Heads | `sashar/models/heads.py` |
| TCBL Module | `sashar/models/tcbl.py` |
| Training Script | `scripts/train.py` |
| Mathematical Framework | `docs/theory/mathematical_framework.md` |
