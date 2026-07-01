# Logistic Regression — Binary & Multinomial Classifier

> A clean, **NumPy-only** implementation of Logistic Regression supporting  
> **binary classification** (sigmoid) and **multinomial classification** (softmax),  
> with **L1 / L2 / ElasticNet regularization** — no scikit-learn under the hood.

---

## Table of Contents

1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [The Model](#2-the-model)
3. [Loss Functions](#3-loss-functions)
4. [Gradient Descent & Update Rules](#4-gradient-descent--update-rules)
5. [Regularization](#5-regularization)
6. [Geometric Intuition](#6-geometric-intuition)
7. [Activation Functions](#7-activation-functions)
8. [Training Pipeline](#8-training-pipeline)
9. [Weight Architecture](#9-weight-architecture)
10. [One-Hot Encoding](#10-one-hot-encoding)
11. [Decision Boundary](#11-decision-boundary)
12. [Gradient Descent Convergence](#12-gradient-descent-convergence)
13. [Penalty Shapes](#13-penalty-shapes)
14. [Effect of C on Weights](#14-effect-of-c-on-weights)
15. [Boundary vs C](#15-boundary-vs-c)
16. [L1 Sparsity](#16-l1-sparsity)
17. [ElasticNet l1_ratio Effect](#17-elasticnet-l1_ratio-effect)
18. [Usage](#18-usage)
19. [Assumptions](#19-assumptions)

---

## 1. What is Logistic Regression?

Logistic Regression models the **probability that a sample belongs to a class** using a linear combination of features passed through a squashing function.

Despite the name, it is a **classification** algorithm — not regression.

| Symbol | Name | Meaning |
|--------|------|---------|
| $\mathbf{w}$ | Weights | Feature coefficients — learned during training |
| $b$ | Bias | Intercept — never regularized |
| $z$ | Logit | Raw score before activation: $z = \mathbf{w}^T\mathbf{x} + b$ |
| $\hat{y}$ | Prediction | Probability output after activation |
| $C$ | Inverse regularization | Larger = weaker penalty; $\lambda = 1/(C \cdot n)$ |
| `l1_ratio` | ElasticNet mix | 0 = pure L2, 1 = pure L1 |

Two modes controlled by `multi_class`:

| Mode | Parameter | Activation | Use case |
|------|-----------|------------|----------|
| Binary / OvR | `'ovr'` | Sigmoid | 2-class problems |
| Multinomial | `'multinomial'` | Softmax | 3+ class problems |

---

## 2. The Model

### Binary

$$z = \mathbf{w}^T\mathbf{x} + b, \qquad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

Output is interpreted as $P(y=1 \mid \mathbf{x})$. Decision: predict class 1 if $\hat{y} \geq 0.5$.

### Multinomial

For $K$ classes, the model produces a score per class:

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + b, \qquad \hat{\mathbf{Y}} = \text{softmax}(\mathbf{Z}), \qquad \mathbf{W} \in \mathbb{R}^{p \times K}$$

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

The `max` subtraction is for numerical stability — it cancels in numerator and denominator but prevents overflow. Prediction: $\hat{y} = \arg\max_k \hat{\mathbf{Y}}_k$.

---

## 3. Loss Functions

### Binary — Cross-Entropy

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

Gradient w.r.t. $\mathbf{w}$ (via chain rule through $\sigma$):

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = -\frac{1}{n}\mathbf{X}^T(y - \hat{y})$$

Update rule — **gradient ascent** on log-likelihood:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \frac{1}{n}\mathbf{X}^T(y - \hat{y})$$

### Multinomial — Categorical Cross-Entropy

$$\mathcal{L}(\mathbf{W}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} Y_{ik}\log\hat{Y}_{ik}$$

Gradient w.r.t. $\mathbf{W}$ (softmax + cross-entropy combine cleanly):

$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{Y}} - \mathbf{Y})$$

Update rule — **gradient descent** on cross-entropy loss:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{Y}} - \mathbf{Y})$$

> The sign difference between binary and multinomial is purely formulation — binary uses log-likelihood ascent, multinomial uses loss descent. Both converge to the same solution.

---

## 4. Gradient Descent & Update Rules

| | Binary | Multinomial |
|---|---|---|
| Logit | $z = \mathbf{X}\mathbf{w} + b$ | $\mathbf{Z} = \mathbf{X}\mathbf{W} + b$ |
| Prediction | $\hat{y} = \sigma(z)$ | $\hat{\mathbf{Y}} = \text{softmax}(\mathbf{Z})$ |
| Error | $y - \hat{y}$ | $\hat{\mathbf{Y}} - \mathbf{Y}$ |
| Weight gradient | $-(1/n)\mathbf{X}^T(y - \hat{y})$ | $(1/n)\mathbf{X}^T(\hat{\mathbf{Y}} - \mathbf{Y})$ |
| Weight update | $\mathbf{w} \mathrel{+}= \eta \cdot (-\partial\mathcal{L}/\partial\mathbf{w})$ | $\mathbf{W} \mathrel{-}= \eta \cdot \partial\mathcal{L}/\partial\mathbf{W}$ |

---

## 5. Regularization

Regularization adds a penalty $R(\mathbf{w})$ to the loss to discourage large weights.

Effective penalty strength: $\lambda = \dfrac{1}{C \cdot n}$ — larger $C$ = weaker penalty.

| Penalty | $R(\mathbf{w})$ | $\nabla R(\mathbf{w})$ | Effect |
|---------|----------------|----------------------|--------|
| `none` | — | $0$ | No shrinkage |
| `l2` | $\frac{\lambda}{2}\|\mathbf{w}\|_2^2$ | $\lambda\mathbf{w}$ | Smooth shrinkage toward zero |
| `l1` | $\lambda\|\mathbf{w}\|_1$ | $\lambda\,\text{sign}(\mathbf{w})$ | Exact zeros — sparse solution |
| `elasticnet` | $\lambda[\rho\|\mathbf{w}\|_1 + \frac{1-\rho}{2}\|\mathbf{w}\|_2^2]$ | $\lambda[\rho\,\text{sign}(\mathbf{w}) + (1-\rho)\mathbf{w}]$ | Both effects |

The bias $b$ is **never penalized** — standard practice, as regularizing it introduces statistical bias without reducing variance.

---

## 6. Geometric Intuition

Logistic Regression learns **linear decision boundaries** — a hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ in feature space.

- **Binary:** one hyperplane separates the two classes.
- **Multinomial:** one hyperplane per class pair — predicted class is $\arg\max_k(\mathbf{X}\mathbf{W} + b)_k$.

Regularization constrains the weight vector:

- **L2 (Ridge):** sphere constraint — smooth, no exact zeros.
- **L1 (Lasso):** diamond constraint — corners on axes, produces exact zeros.
- **ElasticNet:** rounded diamond — combines both effects.

---

## 7. Activation Functions

![Activation Functions](01_activation_functions.png)

**Left — Sigmoid:** smoothly maps any logit to $(0,1)$. Decision boundary at $z=0$ where $\sigma(0)=0.5$.

**Right — Softmax (K=3):** three probability curves always sum to 1. The dominant class probability grows as its logit increases relative to others.

---

## 8. Training Pipeline

![Training Pipeline](02_training_pipeline.png)

`fit()` dispatches based on `multi_class`:

- `'ovr'` → `_fit_binary()` — prepends bias column, runs sigmoid GD loop
- `'multinomial'` → `_fit_multinomial()` — one-hot encodes targets, runs softmax GD loop

Both paths run for `n_iterations` steps and store learned `weights` and `bias`.

---

## 9. Weight Architecture

![Weight Shapes](03_weight_shapes.png)

| Mode | `weights` shape | `bias` shape | Output |
|------|----------------|-------------|--------|
| Binary (OvR) | `(n_features,)` | scalar | $\hat{y} \in (0,1)$ |
| Multinomial | `(n_features, n_classes)` | `(n_classes,)` | $[\,p_0, p_1, \ldots, p_K\,]$ sums to 1 |

In binary mode, bias is folded into the weight vector via a prepended 1s column during training and extracted afterward.

---

## 10. One-Hot Encoding

![One-Hot Encoding](05_one_hot_encoding.png)

Multinomial training requires targets as a matrix $\mathbf{Y} \in \{0,1\}^{n \times K}$:

$$y = [0, 2, 1] \;\rightarrow\; \mathbf{Y} = \begin{bmatrix}1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0\end{bmatrix}$$

Each row sums to 1. The gradient $E = \hat{\mathbf{Y}} - \mathbf{Y}$ has shape $(n, K)$, making the weight update a single matrix multiply: $\partial\mathcal{L}/\partial\mathbf{W} = \mathbf{X}^T E / n$.

The implementation accepts **arbitrary label types** (integers or strings) via `np.unique`. After training, `self.classes_[argmax]` maps predicted indices back to original labels.

---

## 11. Decision Boundary

![Multinomial Decision Boundary](06_decision_boundary.png)

On the Iris dataset (petal length × petal width, standardised):

- **Setosa** is linearly separable from the other two classes.
- **Versicolor / Virginica** overlap slightly — logistic regression draws the best linear boundary possible.
- Stronger regularization (smaller $C$) produces smoother, less overfit boundaries.

---

## 12. Gradient Descent Convergence

![Gradient Descent Convergence](04_gradient_descent.png)

**Left — effect of learning rate `lr`:**

| `lr` | Behaviour |
|------|-----------|
| `0.3` | Smooth, fast convergence |
| `0.05` | Correct but slow |
| `0.01` | Very slow — needs more iterations |
| `1.5` | Overshoots — oscillates and diverges |

**Right — effect of `n_iterations`:** too few iterations leave the model undertrained even with a good learning rate. Monitor loss convergence and increase `n_iterations` if still descending.

---

## 13. Penalty Shapes

![Regularization Penalty Shapes](07_penalty_shapes.png)

Three panels showing the penalty $R(w)$ and its gradient $\nabla R(w)$ for each regularization type:

- **L2:** smooth parabola — gradient grows proportionally, resulting in uniform shrinkage.
- **L1:** V-shape — constant gradient $\pm\lambda$ regardless of weight magnitude, driving small weights to exactly zero.
- **ElasticNet:** between L1 and L2 — blended curve controlled by `l1_ratio`.

---

## 14. Effect of C on Weights

![Effect of C on Weights](08_C_effect_on_weights.png)

As $C$ decreases (stronger regularization), all weights shrink toward zero.

- **L2 (left):** all weights shrink smoothly — shape preserved, magnitude reduced.
- **L1 (right):** some weights collapse to exactly zero — sparse solution emerges at small $C$.

---

## 15. Boundary vs C

![Decision Boundary vs C](09_boundary_vs_C.png)

Four panels showing how the decision boundary changes as $C$ increases (left → right):

| $C$ | Effect |
|-----|--------|
| `0.01` | Very strong regularization — boundaries heavily constrained |
| `0.1` | Moderate — clean, well-generalised separation |
| `1.0` | Default — good balance |
| `10.0` | Weak — boundary closely follows training data, risk of overfit |

---

## 16. L1 Sparsity

![L1 Sparsity](10_l1_sparsity.png)

**Left:** number of zero weights vs $C$ with L1 penalty. Smaller $C$ = stronger L1 push = more exact zeros.

**Right:** weight distribution under no regularization, L1 $C=1.0$, and L1 $C=0.1$. L1 concentrates weights at zero — the distribution becomes increasingly sparse as $C$ decreases.

---

## 17. ElasticNet l1_ratio Effect

![ElasticNet l1_ratio Effect](11_elasticnet_ratio.png)

**Left:** weight values for first 10 features across five `l1_ratio` settings. As `l1_ratio` increases from 0 (pure L2) to 1 (pure L1), weights change shape and some are pushed to zero.

**Right:** number of zero weights vs `l1_ratio`. Zero count rises as `l1_ratio` increases past ~0.4 — the L1 component begins driving features to exact zero.

---

## 18. Usage

### Binary classification — L2 (default)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegressor import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_iterations=1000, penalty='l2', C=1.0)
model.fit(X_train, y_train)

print(f"Accuracy : {model.score(X_test, y_test):.4f}")
print(f"Weights  : {model.weights.shape}")   # (30,)
print(f"Bias     : {model.bias:.4f}")
```

### Binary — L1 (sparse weights)

```python
model = LogisticRegression(lr=0.1, n_iterations=1000, penalty='l1', C=0.5)
model.fit(X_train, y_train)

print(f"Accuracy     : {model.score(X_test, y_test):.4f}")
print(f"Zero weights : {(model.weights == 0).sum()}")
```

### Binary — ElasticNet

```python
model = LogisticRegression(
    lr=0.1, n_iterations=1000,
    penalty='elasticnet', C=1.0, l1_ratio=0.3
)
model.fit(X_train, y_train)
print(f"Accuracy : {model.score(X_test, y_test):.4f}")
```

### Multinomial — Softmax + L2

```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(
    lr=0.1, n_iterations=1000,
    multi_class='multinomial', penalty='l2', C=1.0
)
model.fit(X_train, y_train)

print(f"Accuracy : {model.score(X_test, y_test):.4f}")
print(f"Weights  : {model.weights.shape}")   # (4, 3)
print(f"Bias     : {model.bias.shape}")      # (3,)

probs = model.predict_proba(X_test[:3])
print(probs)
```

### No regularization

```python
model = LogisticRegression(penalty='none')
model.fit(X_train, y_train)
```

---

## 19. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **Linearity** — log-odds are linearly related to features | Decision boundary plot |
| 2 | **No perfect multicollinearity** | Correlation matrix |
| 3 | **Feature scaling** — strongly recommended | Apply `StandardScaler` before fitting |
| 4 | **Independent observations** | Data collection process |
| 5 | **Sufficient sample size** | Rule of thumb: 10+ samples per feature |

> **Feature scaling is essential** — unscaled features create ill-conditioned loss surfaces where gradient descent oscillates. Always apply `StandardScaler`.

> **Bias is never regularized** — penalizing the intercept shifts the mean prediction without reducing variance, which is almost never desired.

---

## Binary vs Multinomial

| Aspect | Binary (`ovr`) | Multinomial (`softmax`) |
|--------|---------------|------------------------|
| Activation | Sigmoid — scalar prob | Softmax — probability vector |
| `weights` shape | `(n_features,)` | `(n_features, n_classes)` |
| `bias` shape | scalar | `(n_classes,)` |
| Target encoding | Raw `{0, 1}` labels | One-hot matrix `(n, K)` |
| Loss | Binary cross-entropy | Categorical cross-entropy |
| Update sign | Ascent on log-likelihood | Descent on cross-entropy |
| Prediction | Threshold at 0.5 | argmax over class scores |
| Works for K > 2? | No | Yes |

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for plots only
sklearn              # optional — for datasets and StandardScaler only
```

---

## License

MIT