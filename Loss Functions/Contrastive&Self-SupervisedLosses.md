Okay, here's a cheat sheet for Contrastive & Self-Supervised Learning (SSL) loss functions, designed for interview preparation. These are crucial for learning representations without explicit human labels.

---

# Contrastive & Self-Supervised Learning (SSL) Losses: Interview Cheat Sheet

**Core Idea:** SSL aims to learn useful data representations by defining pretext tasks that leverage the inherent structure of the data itself. Contrastive learning is a dominant paradigm, where the model learns to pull "similar" (positive) samples together in embedding space and push "dissimilar" (negative) samples apart.

**General Notation:**
*   $x, x', x_i, x_j$: Input data samples or augmented views of the same sample.
*   $f(\cdot)$: Encoder network that maps input $x$ to an embedding vector $z = f(x)$.
*   $g(\cdot)$: Projection head (often an MLP) that maps $z$ to a projected embedding $v = g(z)$, used for loss calculation.
*   $z, z_i, z_j$: Embeddings from the encoder $f$.
*   $v, v_i, v_j$: Projected embeddings from $g$.
*   $\text{sim}(v_i, v_j)$: Cosine similarity between $v_i$ and $v_j$, $\text{sim}(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}$.
*   $\tau$: Temperature hyperparameter (for scaling similarities).
*   $m$: Margin hyperparameter.
*   $B$: Batch size.

---

## I. Classic Contrastive Losses (Revisited in SSL Context)

### 1. Contrastive Loss

*   **Context:** Already covered under Ranking Losses, but fundamental. In SSL, "similar" often means two different augmentations of the *same* image, and "dissimilar" means augmentations of *different* images.
*   **Input:** A pair of projected embeddings $(v_i, v_j)$ and a label $Y \in \{0, 1\}$. $Y=0$ if $v_i, v_j$ form a positive pair (e.g., from same original image), $Y=1$ if negative pair.
*   **Formula (using squared Euclidean distance $D_{ij}^2 = \|v_i - v_j\|_2^2$):**
    $$ L = (1-Y) \frac{1}{2} D_{ij}^2 + Y \frac{1}{2} \max(0, m^2 - D_{ij}^2) $$
*   **Goal:** Positive pairs close, negative pairs far apart by margin $m$.
*   **Pros:** Simple, foundational.
*   **Cons:** Requires careful negative sampling; performance sensitive to margin $m$.

---

### 2. Triplet Loss

*   **Context:** Also covered under Ranking. In SSL, an "anchor" $v_a$ and "positive" $v_p$ are augmentations of the same image. A "negative" $v_n$ is an augmentation of a different image.
*   **Input:** A triplet of projected embeddings $(v_a, v_p, v_n)$.
*   **Formula (using squared Euclidean distance):**
    $$ L = \max(0, \|v_a - v_p\|_2^2 - \|v_a - v_n\|_2^2 + m) $$
*   **Goal:** Anchor closer to positive than to negative by margin $m$.
*   **Pros:** More direct relative comparison than contrastive loss.
*   **Cons:** Triplet mining (finding informative hard negatives) is crucial and can be complex.

---

## II. Modern Self-Supervised Contrastive Losses

### 3. NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy Loss)

*   **Paper:** SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)
*   **Core Idea:** For each augmented sample in a batch, treat other augmentations of the *same* original image as positives, and *all other* augmented samples in the batch as negatives.
*   **Input:** A batch of $2B$ augmented samples, where $(v_i, v_j)$ are a positive pair (from the same original image $x_k$). All other $2(B-1)$ samples in the batch are negatives for $v_i$.
*   **Formula (for one positive pair $(v_i, v_j)$):**
    $$ \ell(i,j) = -\log \frac{\exp(\text{sim}(v_i, v_j)/\tau)}{\sum_{k=1, k \neq i}^{2B} \exp(\text{sim}(v_i, v_k)/\tau)} $$
    The total loss is averaged over all positive pairs in the batch (each sample $v_i$ forms a pair with its corresponding positive $v_j$, and vice-versa, so it's calculated for $v_i$ w.r.t. $v_j$ and for $v_j$ w.r.t. $v_i$).
*   **Description:** Similar to a softmax cross-entropy loss. The model tries to maximize the similarity of positive pairs relative to all negative pairs within the batch. The temperature $\tau$ controls the sharpness of the distribution.
*   **Pros:**
    *   Simple and effective, strong performance.
    *   Does not require explicit negative sampling beyond the batch.
*   **Cons:**
    *   Performance heavily depends on large batch sizes (to have enough negatives).
    *   Sensitive to data augmentation strategies.
*   **When to Use:** A very popular and strong baseline for contrastive SSL.

---

## III. SSL Losses Avoiding Explicit Negative Pairs

These methods aim to learn good representations without directly contrasting against explicit negative samples, addressing some limitations of NT-Xent.

### 4. Barlow Twins Loss

*   **Paper:** Barlow Twins: Self-Supervised Learning via Redundancy Reduction
*   **Core Idea:** Make the cross-correlation matrix between the embeddings of two augmented views of an image as close to the identity matrix as possible. This encourages features to be informative (high variance along diagonal) and decorrelated (low covariance off-diagonal).
*   **Input:** Two batches of projected embeddings $V_A$ and $V_B$ from two augmented views of the same set of original images. $V_A, V_B \in \mathbb{R}^{B \times D}$ (Batch size x Feature dim).
*   **Cross-Correlation Matrix ($C$):**
    $$ C_{ij} = \frac{\sum_{b=1}^{B} (V_A)_{bi} (V_B)_{bj}}{\sqrt{\sum_{b=1}^{B} (V_A)_{bi}^2} \sqrt{\sum_{b=1}^{B} (V_B)_{bj}^2}} $$
    (Normalized features for $V_A, V_B$ are used before computing $C$).
*   **Formula:**
    $$ L_{BT} = \sum_{i=1}^{D} (1 - C_{ii})^2 + \lambda \sum_{i=1}^{D} \sum_{j \neq i} C_{ij}^2 $$
    *   First term (invariance): Pushes diagonal elements $C_{ii}$ to 1 (embeddings of same feature index from different views should be similar).
    *   Second term (redundancy reduction): Pushes off-diagonal elements $C_{ij}$ to 0 (different features should be decorrelated). $\lambda$ is a trade-off hyperparameter.
*   **Pros:**
    *   Does not require negative samples.
    *   Robust to small batch sizes.
    *   Theoretically motivated by information theory (redundancy reduction).
*   **Cons:**
    *   The hyperparameter $\lambda$ needs tuning.
    *   May be sensitive to the dimensionality of the projection head.
*   **When to Use:** When large batch sizes are problematic, or a non-contrastive approach based on information redundancy is desired.

---

### 5. BYOL Loss (Bootstrap Your Own Latent)

*   **Paper:** Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
*   **Core Idea:** Avoids explicit negative pairs by using two neural networks: an "online" network and a "target" network. The online network learns to predict the target network's representation of a different augmented view of the same image. The target network is a slow-moving average (EMA) of the online network's weights (not updated by backprop directly from this loss).
*   **Input:** Two augmented views $x$ and $x'$.
    *   Online network: $v_{\theta} = g_{\theta}(f_{\theta}(x))$ and $q_{\theta}(v_{\theta})$ (predictor MLP).
    *   Target network: $v'_{\xi} = g_{\xi}(f_{\xi}(x'))$ (EMA of online network, $\xi$ is EMA of $\theta$).
*   **Formula (MSE, after L2 normalization of $q_{\theta}(v_{\theta})$ and $v'_{\xi}$):**
    $$ L_{BYOL} = \| \text{normalize}(q_{\theta}(v_{\theta})) - \text{normalize}(v'_{\xi}) \|_2^2 $$
    Symmetrized by computing loss for $(x, x')$ and $(x', x)$ and summing.
*   **Key Elements:** Stop-gradient on target network's output, EMA update for target network weights, predictor MLP in online network.
*   **Pros:**
    *   No negative samples needed, very robust to batch size.
    *   Achieves strong performance.
*   **Cons:**
    *   Relies on EMA and predictor, adding architectural complexity.
    *   Understanding *why* it avoids collapse without negatives is subtle (hypothesized: predictor + EMA prevent learning trivial solutions).
*   **When to Use:** Effective when negative sampling is undesirable or difficult, or when stable training across batch sizes is needed.

---

### 6. SwAV Loss (Swapping Assignments between multiple Views)

*   **Paper:** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
*   **Core Idea:** Enforces consistency of cluster assignments between different views of the same image. It's an online clustering-based approach.
*   **Input:** Multiple ($K \ge 2$) augmented views of images. A set of $C$ learnable "prototype" vectors (cluster centers).
*   **Process:**
    1. For each view $v_i$, compute its similarity to all $C$ prototypes.
    2. Compute an "assignment code" $q_i$ (soft assignment) for $v_i$ by mapping its similarity to prototypes (e.g., using sharpened softmax over prototypes).
    3. **Swapped Prediction Problem:** The loss for view $v_i$ involves predicting its assignment code $q_i$ using a different view $v_j$ of the same image.
*   **Formula (for one pair of views $(v_1, v_2)$ from the same image, and their codes $q_1, q_2$):**
    $$ L(v_1, v_2) = \ell(v_1, q_2) + \ell(v_2, q_1) $$
    where $\ell(v, q) = - \sum_c q_c \log p_c(v)$, and $p_c(v) = \frac{\exp(\frac{1}{\tau} v^T \text{prototype}_c)}{\sum_{c'} \exp(\frac{1}{\tau} v^T \text{prototype}_{c'})}$.
    (Effectively, cross-entropy between the code of one view and the predicted distribution over prototypes for another view).
*   **Pros:**
    *   Does not directly use pairwise negative samples from the batch.
    *   Can work well with smaller batch sizes than SimCLR.
    *   Handles multi-crop augmentations efficiently.
*   **Cons:**
    *   Introduces prototypes that need to be managed and updated (e.g., using Sinkhorn-Knopp algorithm for balanced assignments).
    *   More complex than some other methods.
*   **When to Use:** When an online clustering approach is desired, and multi-crop augmentation strategies are employed.

---

## Key Takeaways for Interviews:

*   **SSL Goal:** Learn meaningful representations from unlabeled data.
*   **Contrastive Learning (SimCLR, Triplet, Contrastive):**
    *   Key idea: Attract positives, repel negatives.
    *   **NT-Xent (SimCLR):** Uses in-batch negatives, needs large batches. Temperature scaling is important.
*   **Non-Contrastive / Asymmetric Networks (BYOL):**
    *   Avoids explicit negatives using architectural designs (online/target networks, EMA, predictor).
    *   Often more robust to batch size. "Collapse" prevention is a key design consideration.
*   **Redundancy Reduction (Barlow Twins):**
    *   Focuses on making the cross-correlation matrix of embeddings from different views close to identity.
*   **Clustering-based (SwAV):**
    *   Learns by enforcing consistent cluster assignments across views. Introduces prototypes.
*   **Common Themes:**
    *   **Data Augmentation:** Critical for creating positive pairs and providing diverse views.
    *   **Projection Head:** An MLP typically added after the encoder, features from which are used for loss computation. Representations from the encoder (before projection head) are used for downstream tasks.
    *   **Temperature ($\tau$):** Used in softmax-based losses (NT-Xent, SwAV) to control the sharpness of the distribution.
*   Be ready to discuss the **motivation** behind moving away from explicit negative sampling (e.g., large batch dependency, quality of negatives) and how different methods achieve this.

Good luck!