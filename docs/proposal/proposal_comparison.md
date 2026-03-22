# PhD Proposal Comparison: DOCX vs. Markdown

This document presents a comprehensive comparison between two iterations of Mohammed Jasim's research proposal.

## 1. Document Overview

| Feature | `HAR Proposal 2025 (3).docx` | [phd_proposal.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) |
| :--- | :--- | :--- |
| **Title** | Enhanced deep learning model for Human Action Recognition (HAR) | Self-Supervised Attention-Based Temporal Segmentation for Human Activity Recognition on Edge Devices |
| **Framework Name** | Nano-HAR | SAS-HAR |
| **Proposed Duration**| 1.5 Years (18 Months) | 4 Years (48 Months) |
| **Level of Detail** | High-level overview, general phases. | Highly detailed, includes pseudocode, specific experiments, ablation studies, and exact metrics. |
| **Key Research Gap** | Segmentation bottleneck & Efficiency/Privacy. | Adds **Label-Efficient Learning (Self-Supervised)** to the existing gaps. |

## 2. Methodology & Architecture Differences

### Similarities
Both proposals share the same foundational motivation to overcome the limitations of the **Fixed-size Sliding Window (FSW)** segmentation method, previously explored by Baraka & Mohd Noor. Both propose a **Dynamic Attention-based Segmentation** mechanism, coupled with a **Hybrid CNN-Transformer** core model, mapped into a deployment layer focused on **NanoML** (Knowledge Distillation, Quantization to INT8) to address the accuracy-efficiency-privacy trilemma.

### Key Divergences

#### A. Introduction of Self-Supervised Learning (SSL)
The most significant methodological addition in the [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) proposal is the new **Self-Supervised Learning Module** (Temporal Contrastive Boundary Learning). 
- The `.docx` model assumes standard supervised learning.
- The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) framework introduces pretext tasks like Contrastive Learning and Continuity Prediction to discover meaningful boundaries from unlabeled continuous sensor streams. This addresses the "Annotation Bottleneck".

#### B. Architectural Specificity
- **Input & Core Encoders:** The `.docx` just mentions "simplified CNN" and "ViT block". The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) specifies Depthwise Separable Convolutions and a lightweight linear Attention Temporal module to model long-range dependencies efficiently.
- **Transitional Activities:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) introduces a specialized component, the **Transitional Activity Specialization Module (TASM)**, targeting a specific >94% F1 score for transitioning activities.

## 3. Structural and Academic Enhancements

The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) file is structured much more like a formal, complete PhD proposal:

- **Research Questions & Gap Analysis:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) explicitly formulates RQ0 to RQ5 and maps them to a structured Research Gap Analysis (with novelty scores mapped out of 10).
- **Novel Contributions:** It names its contributions formally (e.g., TCBL, ABSS, JSCO, TASM) and projects their specific impact (e.g., "-8% boundary F1" in ablation studies).
- **Experimental Protocol:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) lists the exact datasets (WISDM, UCI-HAR, PAMAP2, Opportunity), standard evaluation metrics, 5 specific experiments, and comprehensive ablation studies. The `.docx` simply groups this into a short "Phase 4: Evaluating and Validating" step.
- **Publication Plan:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) maps out three target papers, target venues (TBME, NeurIPS/ICML, MLSys/tinyML), projected impact factors, and expected timelines.

## Summary Conclusion

The `.docx` document serves well as a preliminary **Master's proposal or an early-stage PhD concept**, focusing primarily on replacing the fixed segmentation window and targeting edge devices (NanoML). 

The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) document represents a **mature, full-scale 4-year PhD proposal**. It significantly evolved the core methodology by introducing self-supervised learning for unlabelled boundary detection, and fleshed out the entire program with academic rigor (RQs, datasets, experiments, architecture design, and publication targets).

---

## 4. Addressing Reviewer Comments from DOCX

The `.docx` contained numerous comments left by reviewer **Abdulrahman Baraka**. It is evident that these comments drove the significant improvements seen in the [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) proposal. Here is an analysis of how his major feedback points were implemented:

### Title & Abstract Revisions
- **Comment:** *"Very broad title. Select a focused one."*
- **Addressed ✅:** The generic "Enhanced deep learning model for HAR" was changed to the highly specific "Self-Supervised Attention-Based Temporal Segmentation for Human Activity Recognition on Edge Devices".
- **Comment:** *"Needs more organization to clarify the proposed methodology. The abstract should reflect the gap and your contributions... nothing here refers to the limitations of transitional activities."*
- **Addressed ✅:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) clearly restructured the introduction into sections emphasizing gaps like "The Segmentation Bottleneck", explicitly detailing the >30% misclassification rate for transitional activities like sit-to-stand. 

### Methodology Positioning
- **Comment:** *"As I understand, your segmentation method is novel, not an updated model. Thus, do not refer to the original research, and generalize your work."*
- **Addressed ✅:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) now confidently frames its own architecture (SAS-HAR) as a 9.5/10 novel contribution under "Temporal Contrastive Boundary Learning", distancing itself from being just a minor tweak of the preceding "Deep Similarity Segmentation" work.
- **Comment:** *"There are many methods and approaches to overcome the FSW limitations not only similarity-based. Thus, show and explain them."*
- **Addressed ✅:** The Research Gap Analysis and literature sections were widely expanded in the [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) to cover patch-based transformers (P2LHAP) and lightweight adaptations (TinierHAR).

### Terminology and Specificity
- **Comment:** *"Unify the terminology. Use local features instead of spatial features."*
- **Addressed ✅:** Section `6.2 Component 1` specifically indicates the goal is to "Extract local spatial features".
- **Comment:** *"Lightweight what? Framework, algorithm, model or what?"*
- **Addressed ✅:** The terminology was universally solidified around the "SAS-HAR Framework" comprising specific "Algorithms" and "Modules". 

### Structural Enhancements
- **Comment:** *"Create a graph to describe your proposed methodology" / "It’s better to create a section with a graph to show and explain the HAR stages."*
- **Addressed ✅:** A comprehensive system architecture graph mapping out all stages is now included in section 6.1 (System Overview).
- **Comment:** *"The objectives should be linked with problem statement."*
- **Addressed ✅:** The Research Questions (RQ0-RQ5) flawlessly align with corresponding Hypotheses and Research Objectives (1-4).
- **Comment:** *"Explain each metric with its equation, if any."*
- **Partially Addressed ⚠️:** The [.md](file:///c:/Work/OpenCode/PhDYasmeen/PhD-HAR-Segmentation/docs/proposal/phd_proposal.md) extensively lists the required Evaluation Metrics categorizing them into Segmentation, Classification, and Efficiency metrics, but stops short of including the formal mathematical equations.
