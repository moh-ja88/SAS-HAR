# Mathematical Framework for SAS-HAR

## 1. Problem Formulation

### 1.1 Human Activity Recognition as Temporal Segmentation

**Definition 1.1 (Sensor Stream):** Let $\mathcal{X} = \{x_1, x_2, \ldots, x_T\}$ denote a multivariate time series from wearable sensors, where $x_t \in \mathbb{R}^C$ represents sensor readings from $C$ channels at time step $t$.

**Definition 1.2 (Activity Sequence):** An activity sequence is defined as a set of $N$ contiguous segments:

$$\mathcal{S} = \{(s_i, e_i, y_i)\}_{i=1}^{N}$$

where:
- $s_i \in \{1, \ldots, T\}$ is the start index of segment $i$
- $e_i \in \{1, \ldots, T\}$ is the end index of segment $i$  
- $y_i \in \mathcal{Y} = \{1, \ldots, K\}$ is the activity label
- $s_1 = 1$, $e_N = T$, and $s_{i+1} = e_i + 1$ for all $i$

**Definition 1.3 (Boundary Set):** The boundary set $\mathcal{B} = \{b_1, b_2, \ldots, b_{N-1}\}$ defines transition points between activities, where $b_i = e_i = s_{i+1} - 1$.

### 1.2 Joint Segmentation-Recognition Problem

**Problem 1.1:** Given sensor stream $\mathcal{X}$, find optimal segmentation $\mathcal{S}^*$ and activity labels that minimize:

$$\mathcal{S}^* = \arg\min_{\mathcal{S}} \left[ \underbrace{\mathcal{L}_{seg}(\mathcal{X}, \mathcal{B})}_{\text{Segmentation Loss}} + \lambda \underbrace{\mathcal{L}_{cls}(\mathcal{X}, \mathcal{S})}_{\text{Classification Loss}} \right]$$

where $\lambda$ balances segmentation and classification objectives.

---

## 2. SAS-HAR Architecture

### 2.1 CNN Feature Encoder

The encoder transforms raw sensor data into latent representations:

$$H = f_{enc}(\mathcal{X}; \theta_{enc}) \in \mathbb{R}^{T' \times d}$$

where $T' = T / 2^L$ for $L$ pooling layers.

**Depthwise Separable Convolution:**

$$f_{DSC}(X) = f_{pointwise}(f_{depthwise}(X))$$

where:
- Depthwise: $\text{Conv1d}(X)_c = X_c * k_c$ for each channel $c$
- Pointwise: $\text{Conv1d}(X) = X * W_{1 \times 1}$

**Complexity Analysis:**
- Standard Conv: $\mathcal{O}(C_{in} \cdot C_{out} \cdot K \cdot T)$
- Depthwise Separable: $\mathcal{O}(C_{in} \cdot K \cdot T + C_{in} \cdot C_{out} \cdot T)$
- Reduction factor: $\frac{1}{C_{out}} + \frac{1}{K}$

### 2.2 Efficient Linear Attention Transformer

**Standard Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Complexity: $\mathcal{O}(T^2 \cdot d)$

**Linear Attention (Katharopoulos et al., 2020):**

$$\text{LinearAttn}(Q, K, V) = \phi(Q) \left( \phi(K)^T V \right)$$

where $\phi(\cdot) = \text{elu}(\cdot) + 1$ is a kernel function.

Complexity: $\mathcal{O}(T \cdot d^2)$

**Multi-Head Linear Attention:**

$$\text{MHLinearAttn}(H) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $\text{head}_i = \text{LinearAttn}(H W_i^Q, H W_i^K, H W_i^V)$

### 2.3 Boundary Detection Head

**Semantic Boundary Attention:**

$$\alpha_t = \text{softmax}\left( q_b^T K / \sqrt{d} \right)_t$$

where $q_b \in \mathbb{R}^d$ is a learnable boundary query.

**Boundary Probability:**

$$P(b_t = 1 | H) = \sigma\left( \text{MLP}(H_t) + \alpha_t \cdot \text{Context}(H) \right)$$

**Temporal Smoothing:**

$$\hat{P}(b_t) = \frac{1}{|\mathcal{N}_t|} \sum_{t' \in \mathcal{N}_t} P(b_{t'})$$

where $\mathcal{N}_t = \{t-k, \ldots, t+k\}$ is a local neighborhood.

---

## 3. Self-Supervised Learning (TCBL)

### 3.1 Temporal Contrastive Learning

**Contrastive Objective:**

$$\mathcal{L}_{TC} = -\frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \log \frac{\exp(z_i \cdot z_j^+ / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(z_i \cdot z_k / \tau)}$$

where:
- $\mathcal{P}$ is the set of positive pairs
- $z_i, z_j^+$ are embeddings from augmented views
- $\tau$ is temperature parameter

**Temporal Positive Pairing:**

Positive pairs are defined within same-activity segments:

$$(i, j) \in \mathcal{P} \iff y_i = y_j \land |i - j| \leq \delta$$

### 3.2 Boundary Contrastive Loss

**Objective:** Learn representations where boundaries are distinguishable from non-boundaries.

$$\mathcal{L}_{BC} = -\log \frac{\exp(\text{sim}(h_i, h_j) / \tau)}{\sum_{k \in \mathcal{N}_i} \exp(\text{sim}(h_i, h_k) / \tau)}$$

for $i \in \mathcal{B}$ (boundary), $j \in \mathcal{B}$ (boundary), $k \in \mathcal{X} \setminus \mathcal{B}$ (non-boundary).

### 3.3 Temporal Consistency Loss

**Smoothness within segments:**

$$\mathcal{L}_{smooth} = \sum_{t=1}^{T-1} (1 - b_t) \| f(x_t) - f(x_{t+1}) \|^2$$

**Sharpness at boundaries:**

$$\mathcal{L}_{sharp} = \sum_{t=1}^{T-1} b_t \cdot \max(0, m - \| f(x_t) - f(x_{t+1}) \|)$$

where $m$ is a margin hyperparameter.

### 3.4 Combined TCBL Objective

$$\mathcal{L}_{TCBL} = \mathcal{L}_{TC} + \beta_1 \mathcal{L}_{BC} + \beta_2 \mathcal{L}_{smooth} + \beta_3 \mathcal{L}_{sharp}$$

---

## 4. Training Objectives

### 4.1 Boundary Detection Loss

**Binary Cross-Entropy with Focal Modulation:**

$$\mathcal{L}_{bdry} = -\frac{1}{T} \sum_{t=1}^{T} \alpha_t (1 - p_t)^\gamma \left[ y_t \log(p_t) + (1-y_t) \log(1-p_t) \right]$$

where:
- $p_t = P(b_t = 1 | \mathcal{X})$
- $\alpha_t = \alpha$ if $y_t = 1$, else $1 - \alpha$ (class balancing)
- $\gamma$ is focal parameter

### 4.2 Classification Loss

**Cross-Entropy with Label Smoothing:**

$$\mathcal{L}_{cls} = -\sum_{i=1}^{N} \sum_{k=1}^{K} q_{ik} \log p_{ik}$$

where:
$$q_{ik} = (1 - \epsilon) \cdot \mathbb{1}_{[k = y_i]} + \frac{\epsilon}{K}$$

### 4.3 Joint Loss

$$\mathcal{L}_{total} = \mathcal{L}_{bdry} + \lambda_1 \mathcal{L}_{cls} + \lambda_2 \mathcal{L}_{consist}$$

where $\mathcal{L}_{consist}$ enforces consistency between boundaries and activity changes:

$$\mathcal{L}_{consist} = \sum_{t=1}^{T-1} \mathbb{1}_{[y_t \neq y_{t+1}]} (1 - p_t)$$

---

## 5. Knowledge Distillation

### 5.1 Teacher-Student Framework

**Teacher Model:** $f_T$ with parameters $\theta_T$ (large, ~1.4M params)

**Student Model:** $f_S$ with parameters $\theta_S$ (small, <25K params)

### 5.2 Distillation Loss

**Knowledge Distillation Loss:**

$$\mathcal{L}_{KD} = (1-\alpha) \mathcal{L}_{CE}(y_S, y) + \alpha \mathcal{L}_{KL}(\sigma(z_S/T), \sigma(z_T/T))$$

where:
- $z_S, z_T$ are logits from student and teacher
- $T$ is temperature
- $\sigma$ is softmax

**Feature Distillation:**

$$\mathcal{L}_{feat} = \| f_S(\mathcal{X}) - f_T(\mathcal{X}) \|^2$$

**Attention Transfer:**

$$\mathcal{L}_{attn} = \sum_{l} \| A_S^{(l)} - A_T^{(l)} \|^2_F$$

---

## 6. Evaluation Metrics

### 6.1 Boundary Detection Metrics

**Boundary Precision:**

$$P_B = \frac{|\hat{\mathcal{B}} \cap \mathcal{B}_\tau|}{|\hat{\mathcal{B}}|}$$

**Boundary Recall:**

$$R_B = \frac{|\hat{\mathcal{B}} \cap \mathcal{B}_\tau|}{|\mathcal{B}|}$$

**Boundary F1:**

$$F1_B = \frac{2 \cdot P_B \cdot R_B}{P_B + R_B}$$

where $\mathcal{B}_\tau = \{t : \exists b \in \mathcal{B}, |t - b| \leq \tau\}$ is the tolerance set.

### 6.2 Segmentation Metrics

**Segment IoU:**

$$\text{IoU}_i = \frac{|\hat{s}_i \cap s_i| + |\hat{e}_i \cap e_i|}{|\hat{s}_i \cup s_i| + |\hat{e}_i \cup e_i|}$$

**Edit Distance:**

$$\text{Edit}(\hat{\mathcal{S}}, \mathcal{S}) = \min_{ops} |\{op : op \text{ transforms } \hat{\mathcal{S}} \to \mathcal{S}\}|$$

### 6.3 Classification Metrics

**Accuracy:**

$$\text{Acc} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}_{[\hat{y}_t = y_t]}$$

**Macro F1:**

$$\text{MacroF1} = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

---

## 7. Complexity Analysis

### 7.1 Time Complexity

| Component | Complexity | Parameters |
|-----------|------------|------------|
| CNN Encoder | $\mathcal{O}(C \cdot d \cdot K \cdot T)$ | $\mathcal{O}(C \cdot d \cdot K)$ |
| Linear Attention | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(d^2)$ |
| Boundary Head | $\mathcal{O}(T \cdot d)$ | $\mathcal{O}(d)$ |
| Classification Head | $\mathcal{O}(d \cdot K)$ | $\mathcal{O}(d \cdot K)$ |

**Total:** $\mathcal{O}(T \cdot d^2)$ (linear in sequence length)

### 7.2 Space Complexity

| Component | Memory |
|-----------|--------|
| Activations | $\mathcal{O}(T \cdot d)$ |
| Parameters (Full) | $\sim$1.4M |
| Parameters (Lite) | <25K |

### 7.3 Energy Estimation

$$E_{total} = E_{compute} + E_{memory}$$

$$E_{compute} = \sum_{op} N_{op} \cdot E_{op}$$

where $N_{op}$ is number of operations and $E_{op}$ is energy per operation.

---

## 8. Theoretical Guarantees

### 8.1 Convergence of TCBL

**Theorem 1:** Under mild assumptions, the TCBL objective converges to a local minimum with probability 1 when using stochastic gradient descent with appropriate learning rate scheduling.

**Sketch of Proof:** The loss function is bounded below and Lipschitz continuous. Standard SGD convergence results apply.

### 8.2 Generalization Bound

**Theorem 2:** For the SAS-HAR model with parameters $\theta$, the generalization gap is bounded by:

$$\mathbb{E}[R(\theta) - \hat{R}(\theta)] \leq \mathcal{O}\left(\sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}\right)$$

where $d$ is the effective dimension and $n$ is sample size.

---

## References

1. Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML.
2. Lin, J., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
3. Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." NIPS Workshop.
