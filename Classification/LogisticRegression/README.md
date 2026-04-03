# Logistic Regression — Binary & Multinomial Classifier

> A pure-NumPy implementation of Logistic Regression supporting both **binary classification** (sigmoid) and **multinomial classification** (softmax) from scratch — no scikit-learn under the hood.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Binary Classification (OvR)](#binary-classification-ovr)
- [Multinomial Classification (Softmax)](#multinomial-classification-softmax)
- [Architecture & Class Design](#architecture--class-design)
- [Gradient Descent Optimization](#gradient-descent-optimization)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Key Differences: Binary vs Multinomial](#key-differences-binary-vs-multinomial)

---

## Overview

This module provides a **from-scratch** implementation of Logistic Regression using only NumPy. It supports two modes:

| Mode | Parameter | Activation | Use Case |
|------|-----------|------------|----------|
| Binary / One-vs-Rest | `multi_class='ovr'` | Sigmoid | 2-class problems |
| Multinomial | `multi_class='multinomial'` | Softmax | 3+ class problems |

---

## Mathematical Foundation

### Sigmoid Function (Binary)

Used for binary classification. Maps any real number to the range `(0, 1)`:

```
σ(z) = 1 / (1 + e^(−z))
```

The output is interpreted as **P(y = 1 | X)**.

### Softmax Function (Multinomial)

Generalises sigmoid to K classes. Converts a vector of raw scores into a **probability distribution**:

```
softmax(z_k) = e^(z_k) / Σ e^(z_j)   for j = 1..K
```

Each output is `P(y = class_k | X)`, and all outputs sum to 1.

> **Numerical Stability:** The implementation subtracts `max(z)` before exponentiation to prevent overflow — a standard trick that doesn't change the result mathematically.

---

## Binary Classification (OvR)

### How it works

1. A **bias column** (all 1s) is prepended to `X_train`
2. A single weight vector `w ∈ ℝ^(n_features+1)` is initialized to zeros
3. For each iteration, the model computes:

```
ŷ = σ(X · w)
```

4. Weights are updated via **gradient ascent on log-likelihood**:

```
w ← w + lr × Xᵀ(y − ŷ) / n
```

5. After training, `weights[0]` becomes `self.bias`, and `weights[1:]` become `self.weights`

### Loss Function (implicit)

Binary Cross-Entropy:

```
L = −(1/n) Σ [y log(ŷ) + (1−y) log(1−ŷ)]
```

---

## Multinomial Classification (Softmax)

### How it works

1. Weights are now a **matrix**: `W ∈ ℝ^(n_features × n_classes)`
2. Bias is a **vector**: `b ∈ ℝ^(n_classes)`
3. Labels `y` are **one-hot encoded** into `Y ∈ ℝ^(n_samples × n_classes)`
4. For each iteration:

```
logits = X · W + b          shape: (n_samples, n_classes)
ŷ      = softmax(logits)    shape: (n_samples, n_classes)
error  = ŷ − Y              shape: (n_samples, n_classes)
```

5. Gradients are computed and applied:

```
W ← W − lr × (Xᵀ · error) / n
b ← b − lr × mean(error, axis=0)
```

### Loss Function (implicit)

Categorical Cross-Entropy:

```
L = −(1/n) Σ Σ Y_{ik} log(ŷ_{ik})
```

---

## Architecture & Class Design

```
LogisticRegression
├── __init__(lr, n_iterations, multi_class)
│
├── Core Math
│   ├── sigmoid(z)            — binary activation
│   ├── softmax(z)            — multinomial activation
│   └── _one_hot_encode(y)    — label matrix for softmax
│
├── Training
│   ├── _fit_binary(X, y)     — single weight vector + bias scalar
│   └── _fit_multinomial(X, y)— weight matrix + bias vector
│
└── Inference
    ├── fit(X, y)             — dispatcher → routes to _fit_binary or _fit_multinomial
    ├── predict_proba(X)      — probabilities  (n_samples,) or (n_samples, n_classes)
    ├── predict(X, threshold) — class labels
    └── score(X, y)           — accuracy
```

---

## Gradient Descent Optimization

Both modes use **Batch Gradient Descent** — the full dataset is used to compute each weight update. This is in contrast to Stochastic (SGD) or Mini-Batch variants.

### Update Rule Comparison

| | Binary | Multinomial |
|---|---|---|
| **Prediction** | `ŷ = σ(Xw + b)` | `ŷ = softmax(XW + b)` |
| **Error** | `y − ŷ` (scalar per sample) | `ŷ − Y` (vector per sample) |
| **Weight update** | `w += lr × Xᵀ(y−ŷ)/n` | `W -= lr × Xᵀ(ŷ−Y)/n` |
| **Bias update** | included in w[0] | `b -= lr × mean(ŷ−Y)` |

> Note the **sign flip** between binary and multinomial update rules — binary uses gradient *ascent* on log-likelihood; multinomial uses gradient *descent* on cross-entropy. Both converge to the same solution.

---

## API Reference

### Constructor

```python
LogisticRegression(lr=0.5, n_iterations=2000, multi_class='ovr')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.5 | Learning rate (step size per iteration) |
| `n_iterations` | int | 2000 | Number of gradient descent steps |
| `multi_class` | str | `'ovr'` | `'ovr'` for binary, `'multinomial'` for softmax |

### Methods

#### `fit(X_train, y_train)`
Trains the model. Automatically dispatches to `_fit_binary` or `_fit_multinomial`.

- **X_train** — array-like, shape `(n_samples, n_features)`
- **y_train** — array-like, shape `(n_samples,)` — integer or string labels

#### `predict_proba(X_test)`
Returns class probabilities.

- Binary → shape `(n_samples,)` — probability of class 1
- Multinomial → shape `(n_samples, n_classes)` — probability per class

#### `predict(X_test, threshold=0.5)`
Returns predicted class labels.

- Binary → thresholds `predict_proba` at `threshold`
- Multinomial → returns `argmax` over class probabilities

#### `score(X_test, y_test)`
Returns **accuracy** = fraction of correct predictions.

---

## Usage Examples

### Binary Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_iterations=1000, multi_class='ovr')
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
# Accuracy: ~0.9737

probs = model.predict_proba(X_test[:3])
print(probs)   # [0.9981, 0.0023, 0.9876] — scalar per sample
```

### Multinomial Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_iterations=1000, multi_class='multinomial')
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
# Accuracy: ~0.9667

probs = model.predict_proba(X_test[:3])
print(probs)
# [[0.001, 0.023, 0.976],   <- class 2
#  [0.972, 0.027, 0.001],   <- class 0
#  [0.003, 0.994, 0.003]]   <- class 1
```

---

## Hyperparameter Guide

### Learning Rate (`lr`)

| Value | Effect |
|-------|--------|
| Too high (> 1.0) | Oscillates, may diverge |
| Good range (0.01–0.3) | Smooth convergence |
| Too low (< 0.001) | Very slow convergence |

### Iterations (`n_iterations`)

| Value | Typical use |
|-------|-------------|
| 100–500 | Quick experiments |
| 1000–2000 | Standard training |
| 5000+ | Complex/noisy data |

> **Tip:** Always **standardize** your features (`StandardScaler`) before training. Logistic Regression is sensitive to feature scales — unnormalised features cause the gradient to follow steep, narrow valleys, slowing convergence.

---

## Key Differences: Binary vs Multinomial

| Aspect | Binary (`ovr`) | Multinomial (`softmax`) |
|--------|---------------|------------------------|
| **Activation** | Sigmoid → scalar | Softmax → probability vector |
| **Weight shape** | `(n_features,)` | `(n_features, n_classes)` |
| **Bias shape** | scalar | `(n_classes,)` |
| **Target encoding** | Raw `{0, 1}` labels | One-hot matrix |
| **Loss (implicit)** | Binary cross-entropy | Categorical cross-entropy |
| **Prediction** | Threshold on prob | `argmax` over class probs |
| **Works for K > 2?** | No (requires OvR wrapping) | Yes, natively |

---

## Notes on Implementation Choices

- **No regularization** is applied — adding L2 regularization (`λ * weights`) to the gradient update is a natural extension.
- **Bias handling differs**: the binary path folds bias into the weight vector via a bias column; the multinomial path keeps `W` and `b` separate for clarity.
- **`classes_`** is set at `fit()` time via `np.unique(y)` and is used by `predict()` to map `argmax` indices back to the original label space (works with string labels too).
