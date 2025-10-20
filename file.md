# 🧠 Deep Learning – MLP & RNN – Exam Review


---

## 1️⃣ Foundations

### **Neural Network Basics**

A neural network is a sequence of linear and non-linear transformations:
$$
a^{(l)} = f(z^{(l)}) \quad \text{where} \quad z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

* $a^{(0)} = x$: input
* $W^{(l)}$: weights matrix
* $b^{(l)}$: bias vector
* $f$: activation function (non-linearity)

### **Forward Propagation**

Computes output step-by-step from input:
$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = f(z^{(l)})
$$
Final layer output: $\hat{y} = f_{out}(z^{(L)})$

### **Loss Function**

Measures prediction error:

* Regression: $L = \frac{1}{2}(y - \hat{y})^2$
* Binary classification: $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
* Multi-class: $L = -\sum y_i \log(\hat{y}_i)$, with $\hat{y}_i$ prob given by softmax

---

## 2️⃣ Derivatives and Backpropagation

### **Goal**

Compute gradients $ \frac{\partial L}{\partial W^{(l)}}, \frac{\partial L}{\partial b^{(l)}} $ efficiently.

### **Chain Rule**

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial W^{(l)}}
$$

### **Backward Pass**

Define error at layer (l):
$$
\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)})
$$

Then:
$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T, \quad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

---

## 3️⃣ Optimization Algorithms

### **Gradient Descent**

Update rule:
$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

* **Batch GD**: uses all samples
* **Stochastic GD (SGD)**: 1 sample per step
* **Mini-batch GD**: small group (typical)

### **Variants**

* **Momentum:** $v_t = \beta v_{t-1} + (1-\beta)\nabla_\theta L$
* **RMSProp:** divides by root of past squared gradients
* **Adam:** adaptive combination of Momentum + RMSProp

---

## 4️⃣ Regularization Techniques

| Technique           | Idea                             | Formula / Example               |
| ------------------- | -------------------------------- | ------------------------------- |
| L2 Regularization   | penalize large weights           | $ L = L_0 + \lambda \|W\|^2 $     |
| Dropout             | randomly remove neurons          | deactivate neurons with prob. p |
| Early stopping      | stop training before overfitting | monitor validation loss         |
| Batch normalization | normalize activations            | $ \hat{z} = (z - \mu)/\sigma$    |

---

## 5️⃣ Non-Linearities (Activations)

| Function   | Formula                               | Derivative                 | Notes              |
| ---------- | ------------------------------------- | -------------------------- | ------------------ |
| Sigmoid    | $ \frac{1}{1+e^{-x}} $                | $ \sigma(x)(1-\sigma(x)) $ | Vanishing gradient |
| tanh       | $ \frac{e^x - e^{-x}}{e^x + e^{-x}} $ |$ 1 - \tanh^2(x) $         | Centered around 0  |
| ReLU       | $ \max(0, x) $                        | $ 1_{x>0} $                | Sparse, efficient  |
| Leaky ReLU | $ \max(0.01x, x) $                    | -                          | Fix dead neurons   |

---

## 6️⃣ Weight Initialization

| Method            | Formula                             | Usage        |
| ----------------- | ----------------------------------- | ------------ |
| Xavier (Glorot)   | $(Var(W) = \frac{2}{n_{in}+n_{out}})$ | tanh/sigmoid |
| He initialization | $(Var(W) = \frac{2}{n_{in}})$         | ReLU         |

---

## 7️⃣ Vanishing/Exploding Gradient

* **Problem:** repeated multiplication by small/large values → gradient → 0 or ∞
* **Solutions:**

  * Proper initialization (He, Xavier)
  * BatchNorm
  * Skip connections (ResNet)
  * LSTM/GRU (for RNNs)

---

## 8️⃣ RNN (Recurrent Neural Network)

### **Idea**

Handle sequences — output depends on previous time steps.

$$
h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$
$$
\hat{y}*t = f*{out}(W_{hy}h_t + b_y)
$$

### **Types**

| Type         | Description              |
| ------------ | ------------------------ |
| One-to-one   | basic NN                 |
| One-to-many  | image captioning         |
| Many-to-one  | sentiment classification |
| Many-to-many | translation, tagging     |

---

## 9️⃣ Backpropagation Through Time (BPTT)

* Unroll RNN over time steps
* Apply chain rule backward across time
* Problem: **vanishing gradient through time**
  → solved by LSTM / GRU.

---

## 🔟 LSTM (Long Short-Term Memory)

### **Goal:** preserve long-term dependencies.

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) &\text{(forget gate)}\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) &\text{(input gate)}\
\tilde{C}*t &= \tanh(W_C [h*{t-1}, x_t] + b_C) &\text{(candidate)}\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}*t &\text{(cell state)}\
o_t &= \sigma(W_o [h*{t-1}, x_t] + b_o) &\text{(output gate)}\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

✅ Prevents vanishing gradient with constant error flow via (C_t).

---

## 11️⃣ Encoder–Decoder (Seq2Seq)

Used in **translation**, **summarization**, etc.

* **Encoder:** RNN reads input sequence → context vector.
* **Decoder:** generates output sequence from context.
* Limitation: context vector = bottleneck → solved by **Attention**.

---

## 12️⃣ Attention Mechanism

### **Idea:**

Instead of one context, decoder attends to **all encoder states** with learned weights.

$$
\text{score}(s_t, h_i) = s_t^T W_a h_i
$$
$$
\alpha_i = \text{softmax(score)} \quad ; \quad c_t = \sum_i \alpha_i h_i
$$
Decoder then uses (c_t) to generate next token.

---

## 13️⃣ Self-Attention & Transformer

### **Core formula**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

* (Q): query, (K): key, (V): value (from same sequence)
* Computes relationships between all tokens.

### **Multi-Head Attention**

Multiple parallel attentions → capture different relations.

### **Feed-forward block**

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

### **Add & Norm**

Residual connection + Layer Normalization:
$$
x' = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

---

## 14️⃣ Positional Encoding

Since Transformers lack recurrence:
$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Adds sequence order information to embeddings.

---

## 15️⃣ Common RNN Applications

| Task                               | Input–Output | Notes                        |
| ---------------------------------- | ------------ | ---------------------------- |
| **Language modeling**              | many-to-one  | predict next word            |
| **Text generation**                | many-to-many | sample from softmax output   |
| **NER (Named Entity Recognition)** | many-to-many | classify each token          |
| **Sentiment classification**       | many-to-one  | final hidden → classifier    |
| **Machine translation**            | seq2seq      | encoder-decoder w/ attention |

---

## 16️⃣ Word Representations

### **One-hot encoding**

* Each word = vector with 1 at its index
* High dimension, no semantic meaning

### **Word embeddings**

* Dense vector representation
* Learned to capture similarity (Word2Vec, GloVe)
* Dimension ~100–300

---

## 17️⃣ Summary Table

| Concept        | Purpose               | Key Equation                                                    |
| -------------- | --------------------- | --------------------------------------------------------------- |
| Forward pass   | compute outputs       | $(a=f(Wx+b))$                                                     |
| Loss           | measure error         | $(L=\frac{1}{2}(y-\hat{y})^2)$                                    |
| Backprop       | compute gradients     | $(\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)}))$ |
| SGD            | optimize parameters   | $(\theta\leftarrow\theta - \eta\nabla_\theta L)$                  |
| Regularization | avoid overfitting     | Dropout, L2, BatchNorm                                          |
| LSTM           | handle long-term deps | gating mechanism                                                |
| Attention      | dynamic context       | $(\text{softmax}(QK^T/\sqrt{d_k})V)$                              |
| Transformer    | parallel seq model    | Self-Attention + FFN + PE                                       |

---

## 18️⃣ Quick Check (Reflection Questions)

* Why do we use non-linearities?
* How does batch-norm help training?
* What happens if weights initialized to 0?
* Why does ReLU cause “dead neurons”?
* How does LSTM fight vanishing gradients?
* Why is attention better than fixed context?
* What is the main computational cost of Transformers?
* Why use embeddings instead of one-hot vectors?


# 🧩 CNN – Convolutional Neural Networks – Exam Review

---

## 1️⃣ Core Idea

CNNs are designed to process data with spatial structure (e.g., images).

Instead of connecting every input neuron to every output neuron (like MLPs), CNNs use **local receptive fields** and **shared weights** to detect local patterns.

---

## 2️⃣ Convolution Operation

$$
z_{i,j}^{(k)} = (X * W^{(k)})_{i,j} + b^{(k)}
$$

* $X$: input image or feature map
* $W^{(k)}$: filter (kernel) for feature map (k)
* $*$: convolution
* $b^{(k)}$: bias

Each filter detects a feature (edges, colors, shapes…).

**Stride:** step size for moving the kernel
**Padding:** add zeros to keep spatial size constant

* *Same padding*: output same size
* *Valid padding*: no padding, smaller output

---

## 3️⃣ Output Size Formula

If input size = $n \times n$, kernel size = $f$, stride = $s$, padding = $p$:

$$
n_{out} = \frac{n - f + 2p}{s} + 1
$$

---

## 4️⃣ Pooling Layer

Reduces spatial dimension, retains important info.

| Type            | Operation                 | Example             |
| --------------- | ------------------------- | ------------------- |
| Max pooling     | takes max value in region | highlights features |
| Average pooling | takes mean value          | smooths activations |

Example: 2×2 max pooling halves height and width.

---

## 5️⃣ Flatten + Fully Connected Layer

After several convolution + pooling blocks, the result is **flattened** into a vector → fed to MLP for classification.

---

## 6️⃣ Common CNN Architectures

| Network       | Innovation                                     |
| ------------- | ---------------------------------------------- |
| **LeNet-5**   | first CNN for digit recognition                |
| **AlexNet**   | ReLU + dropout + GPU training                  |
| **VGG**       | smaller 3×3 filters, deeper                    |
| **ResNet**    | skip connections to prevent vanishing gradient |
| **Inception** | multi-scale filters in parallel                |

---

## 7️⃣ Parameter Sharing Advantage

Each filter is reused across the image → far fewer parameters than an MLP, improves generalization and speed.

---

## 8️⃣ Regularization in CNNs

* **Dropout**: remove neurons randomly
* **Data augmentation**: random crops, rotations, flips
* **Batch normalization**: stabilizes training
* **Weight decay (L2)**

---

## 9️⃣ Backpropagation in CNNs

Gradients flow through:

1. Convolution operation (weight sharing)
2. Non-linearity (ReLU)
3. Pooling (max/avg)
4. Fully connected layers

Important: derivative of convolution = convolution with flipped kernel.

---

## 🔟 CNN Applications

* Image classification (ResNet, VGG)
* Object detection (YOLO, Faster R-CNN)
* Segmentation (U-Net)
* Image captioning (CNN + RNN)
* Feature extraction for transfer learning

---

## 11️⃣ Reflection Questions

* Why do CNNs need less parameters than MLPs?
* What is the role of stride and padding?
* Why use ReLU in CNNs?
* What is learned in each filter?
* Why does ResNet solve the vanishing gradient problem?
