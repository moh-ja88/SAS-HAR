# Research Timeline

## 4-Year PhD Program (48 Months)

---

## Year 1: Foundation (Months 1-12)

### Semester 1 (Months 1-6): Literature Review and Baselines

#### Month 1-2: Comprehensive Literature Review
**Activities:**
- Read 100+ papers on HAR, segmentation, SSL, edge AI
- Create annotated bibliography
- Identify state-of-the-art methods (2022-2026)
- Document research gaps in detail

**Deliverables:**
- [ ] Literature review document (100+ papers)
- [ ] Gap analysis report
- [ ] Weekly progress reports

**Milestone:** Comprehensive understanding of HAR field

---

#### Month 3: Dataset Preparation
**Activities:**
- Download benchmark datasets (WISDM, UCI-HAR, PAMAP2, Opportunity)
- Create unified preprocessing pipeline
- Implement data loaders for PyTorch
- Document dataset characteristics

**Deliverables:**
- [ ] Preprocessing scripts
- [ ] Data loaders
- [ ] Dataset documentation

**Milestone:** All datasets ready for experiments

---

#### Month 4-5: Baseline Reproduction
**Activities:**
- Implement Fixed Sliding Window baseline
- Reproduce Adaptive Sliding Window (Noor 2017)
- Reproduce Statistical Similarity (Baraka 2023)
- Reproduce Deep Similarity Segmentation (Baraka 2024)

**Deliverables:**
- [ ] Baseline implementations
- [ ] Performance reports matching published results (±2%)

**Milestone:** All baselines reproduced successfully

---

#### Month 6: Year 1 Proposal Defense
**Activities:**
- Compile Year 1 findings
- Prepare proposal presentation
- Defend research direction

**Deliverables:**
- [ ] Year 1 annual report
- [ ] Proposal presentation
- [ ] Refined research questions

**Milestone:** ✅ Proposal defended

---

### Semester 2 (Months 7-12): SOTA Analysis and Initial Design

#### Month 7-8: SOTA Implementation
**Activities:**
- Implement P2LHAP (2024) transformer
- Implement TinierHAR (2025) lightweight model
- Implement XTinyHAR (2025) distillation
- Create unified evaluation framework

**Deliverables:**
- [ ] SOTA implementations
- [ ] Evaluation framework
- [ ] Benchmark results

---

#### Month 9-10: Self-Supervised Framework Design
**Activities:**
- Design Temporal Contrastive Boundary Learning (TCBL)
- Implement continuity prediction pretext task
- Design temporal contrastive loss
- Initial experiments on unlabeled data

**Deliverables:**
- [ ] TCBL implementation
- [ ] Initial SSL results
- [ ] Ablation experiments

---

#### Month 11-12: Attention Module Design
**Activities:**
- Design Semantic Boundary Attention (SBA)
- Implement efficient linear attention
- Compare to standard attention
- Initial boundary detection experiments

**Deliverables:**
- [ ] SBA implementation
- [ ] Attention visualization tools
- [ ] Boundary F1 results

**Milestone:** Initial TCBL + SBA results

---

## Year 2: Core Algorithm Development (Months 13-24)

### Semester 3 (Months 13-18): Architecture Design

#### Month 13-15: Hybrid CNN-Transformer
**Activities:**
- Design hybrid architecture
- Implement CNN encoder with depthwise separable conv
- Implement efficient transformer temporal module
- Integrate segmentation and classification

**Deliverables:**
- [ ] Hybrid architecture implementation
- [ ] Parameter efficiency analysis
- [ ] Initial accuracy results

---

#### Month 16-18: Self-Supervised Training Pipeline
**Activities:**
- Complete TCBL implementation
- Combine all pretext tasks
- Train on multiple datasets
- Evaluate label efficiency

**Deliverables:**
- [ ] Complete SSL pipeline
- [ ] Label efficiency curves
- [ ] Transfer learning experiments

**Milestone:** Working SAS-HAR prototype

---

### Semester 4 (Months 19-24): Edge Optimization

#### Month 19-21: Knowledge Distillation
**Activities:**
- Train large teacher model
- Design student architecture (<25K params)
- Implement multi-stage distillation
- Evaluate accuracy-efficiency trade-off

**Deliverables:**
- [ ] Teacher-student training pipeline
- [ ] Distilled student model
- [ ] Compression analysis

---

#### Month 22-24: Edge Deployment
**Activities:**
- Implement quantization-aware training (INT8)
- Deploy on Arduino Nano 33 BLE
- Deploy on STM32F4
- Measure real energy/latency

**Deliverables:**
- [ ] Quantized models
- [ ] Arduino/STM32 deployment
- [ ] Energy/latency measurements

**Milestone:** ✅ Paper 1 draft complete

---

## Year 3: Validation and Optimization (Months 25-36)

### Semester 5 (Months 25-30): Comprehensive Evaluation

#### Month 25-27: Benchmark Evaluation
**Activities:**
- Evaluate on all datasets (WISDM, UCI-HAR, PAMAP2, Opportunity)
- Compare to all baselines
- Document all results
- Statistical significance testing

**Deliverables:**
- [ ] Complete benchmark results
- [ ] Statistical analysis
- [ ] Comparison tables

---

#### Month 28-30: Ablation Studies
**Activities:**
- Ablate TCBL components
- Ablate attention mechanisms
- Ablate transitional module
- Document contribution of each component

**Deliverables:**
- [ ] Ablation study results
- [ ] Component importance analysis
- [ ] Visualization of learned features

---

### Semester 6 (Months 31-36): Real-World Validation

#### Month 31-33: Real-World Deployment Study
**Activities:**
- Recruit 10 participants
- Deploy on smartwatches
- Collect 2 weeks of data per participant
- Measure real-world performance

**Deliverables:**
- [ ] Deployment infrastructure
- [ ] Real-world dataset (optional release)
- [ ] User study results

---

#### Month 34-36: Paper Writing
**Activities:**
- Complete Paper 1 writing
- Begin Paper 2 draft
- Begin Paper 3 draft
- Submit Paper 1 to target venue

**Deliverables:**
- [ ] Paper 1 submitted
- [ ] Paper 2 draft
- [ ] Paper 3 draft

**Milestone:** ✅ Paper 1 submitted

---

## Year 4: Thesis and Dissemination (Months 37-48)

### Semester 7 (Months 37-42): Paper Publications

#### Month 37-39: Paper Revisions and Submissions
**Activities:**
- Address Paper 1 reviewer comments
- Complete Paper 2 writing
- Submit Paper 2
- Complete Paper 3 writing

**Deliverables:**
- [ ] Paper 1 revised
- [ ] Paper 2 submitted
- [ ] Paper 3 complete

---

#### Month 40-42: Thesis Writing (Part 1)
**Activities:**
- Write Chapter 1: Introduction
- Write Chapter 2: Literature Review
- Write Chapter 3: Methodology
- Write Chapter 4: Self-Supervised Learning

**Deliverables:**
- [ ] Thesis chapters 1-4 draft

**Milestone:** ✅ Papers 1-2 submitted/accepted

---

### Semester 8 (Months 43-48): Thesis Completion

#### Month 43-45: Thesis Writing (Part 2)
**Activities:**
- Write Chapter 5: Experiments
- Write Chapter 6: Results and Analysis
- Write Chapter 7: Discussion and Future Work
- Integrate all chapters

**Deliverables:**
- [ ] Thesis complete draft

---

#### Month 46: Thesis Submission
**Activities:**
- Final revisions
- Supervisor review
- Format checking
- Official submission

**Deliverables:**
- [ ] Thesis submitted

---

#### Month 47: Defense Preparation
**Activities:**
- Prepare defense presentation
- Practice defense talk
- Anticipate questions
- Prepare supplementary materials

**Deliverables:**
- [ ] Defense presentation
- [ ] Defense rehearsal

---

#### Month 48: PhD Defense and Open Source Release
**Activities:**
- PhD defense
- Address committee feedback
- Final thesis submission
- Open-source code release on GitHub

**Deliverables:**
- [ ] Successfully defended
- [ ] Final thesis submitted
- [ ] Open-source repository

**Milestone:** ✅ PhD completed

---

## Timeline Visualization

```
Year 1 (Foundation)
├── S1: Literature Review ──────► Dataset Prep ──► Baselines ──► Defense
└── S2: SOTA Implementation ────► SSL Design ────► Attention ──────────►

Year 2 (Development)
├── S3: Hybrid Architecture ────► SSL Pipeline ──────────────────────►
└── S4: Knowledge Distillation ─► Edge Deployment ───────────────────►

Year 3 (Validation)
├── S5: Benchmark Evaluation ───► Ablation Studies ──────────────────►
└── S6: Real-World Study ───────► Paper Writing ─────────────────────►

Year 4 (Thesis)
├── S7: Paper Submissions ──────► Thesis Writing (Part 1) ────────────►
└── S8: Thesis Writing (Part 2) ► Defense ────────► PhD Complete ────►
```

---

## Key Milestones

| Month | Milestone | Success Criteria |
|-------|-----------|------------------|
| 6 | Proposal Defense | Approved by committee |
| 12 | Initial Results | TCBL achieves >80% @ 10% labels |
| 18 | Working Prototype | SAS-HAR achieves >95% accuracy |
| 24 | Edge Deployment | <25K params, <50 nJ energy |
| 30 | Complete Evaluation | All datasets, all baselines |
| 36 | Paper 1 Submitted | IEEE TBME submission |
| 42 | Papers 1-2 Accepted | At least 1 published |
| 48 | PhD Defense | Successfully defended |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Baseline reproduction fails | Medium | High | Contact original authors, extra time buffer |
| SSL doesn't work | Low | High | Fallback to semi-supervised |
| Edge deployment too slow | Medium | Medium | More aggressive compression |
| Paper rejections | High | Medium | Target multiple venues simultaneously |
| Real-world study challenges | Medium | Medium | Start with pilot (3 users) |

---

*Last Updated: March 2026*
