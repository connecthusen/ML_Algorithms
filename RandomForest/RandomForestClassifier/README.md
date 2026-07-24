# Random Forest Classifier — Ensemble of Decision Trees

> A clean, **NumPy-only** implementation of Random Forest Classification.  
> Trains multiple Decision Trees on **bootstrap samples** with **random feature subsets** —  
> final prediction is the **majority vote** across all trees.  
> More robust and accurate than a single tree — reduces overfitting through ensemble averaging.

---

## Table of Contents

1. [What is Random Forest?](#1-what-is-random-forest)
2. [The Model](#2-the-model)
3. [Bootstrap Sampling](#3-bootstrap-sampling)
4. [Random Feature Subsets](#4-random-feature-subsets)
5. [Voting & Prediction](#5-voting--prediction)
6. [Geometric Intuition](#6-geometric-intuition)
7. [Decision Boundary](#7-decision-boundary)
8. [Single Tree vs Forest](#8-single-tree-vs-forest)
9. [Build Pipeline](#9-build-pipeline)
10. [Predicted vs Actual](#10-predicted-vs-actual)
11. [Effect of n_estimators](#11-effect-of-n_estimators)
12. [Feature Importance](#12-feature-importance)
13. [Confusion Matrix & Metrics](#13-confusion-matrix--metrics)
14. [Usage](#14-usage)
15. [Assumptions](#15-assumptions)

---

## 1. What is Random Forest?

Random Forest is an **ensemble** method that trains many Decision Trees independently and combines their predictions via majority vote.

Given $n$ observations $(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)$:

$$\hat{y} = \text{majority\_vote}\!\left(\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(T)}\right)$$

where $\hat{y}^{(t)}$ is the prediction of tree $t$.

| Symbol | Name | Meaning |
|--------|------|---------|
| $T$ | `n_estimators` | Number of trees in the forest |
| $B$ | Bootstrap sample | Random sample with replacement — same size as training set |
| $k$ | `max_features` | Number of features considered at each split |
| `'sqrt'` | Default strategy | $k = \lfloor\sqrt{p}\rfloor$ — standard for classification |
| `'log2'` | Alternative | $k = \lfloor\log_2 p\rfloor$ — more aggressive feature reduction |

Two sources of randomness — what makes each tree different:

| Source | What it does |
|--------|-------------|
| **Bootstrap sampling** | Each tree sees a different random subset of training rows |
| **Random feature subsets** | Each split only considers $k$ random features — not all $p$ |

---

## 2. The Model

Each tree $t$ is trained independently:

1. Draw a **bootstrap sample** $D^{(t)}$ of size $n$ from $D$ with replacement
2. Fit a `DecisionTree` on $D^{(t)}$ using random feature subsets at each split
3. Store the fitted tree in `self.trees_`

At prediction time, each tree votes and the majority wins:

$$\hat{y} = \arg\max_k \sum_{t=1}^{T} \mathbf{1}[\hat{y}^{(t)} = k]$$

Probability estimates use vote fractions:

$$P(y = k \mid \mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[\hat{y}^{(t)} = k]$$

---

## 3. Bootstrap Sampling

Each tree trains on a **bootstrap sample** — $n$ samples drawn with replacement from the original $n$ training points:

$$D^{(t)} = \{(\mathbf{x}_{i_1}, y_{i_1}), \ldots, (\mathbf{x}_{i_n}, y_{i_n})\}, \quad i_j \sim \text{Uniform}(1, n)$$

On average, each bootstrap sample contains about **63.2%** of unique training points — the rest (~36.8%) are repeated. The unused points are called **Out-of-Bag (OOB)** samples and can be used for free validation.

This diversity in training data ensures trees make different errors — and their errors cancel out when voting.

---

## 4. Random Feature Subsets

At every split in every tree, only $k$ randomly chosen features are considered:

| `max_features` | $k$ | When to use |
|---------------|-----|------------|
| `'sqrt'` | $\lfloor\sqrt{p}\rfloor$ | Default — good for most classification problems |
| `'log2'` | $\lfloor\log_2 p\rfloor$ | More aggressive — high-dimensional data |
| `int` | fixed $k$ | Manual control |
| `None` | $p$ (all features) | Equivalent to Bagged Trees — no feature randomness |

This feature randomness **decorrelates** the trees — even if one feature is very strong, different trees will use different features for their splits, reducing variance.

---

## 5. Voting & Prediction

**Hard vote** — `predict()`:

$$\hat{y} = \text{mode}\!\left(\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(T)}\right)$$

**Soft vote** — `predict_proba()`:

$$P(k \mid \mathbf{x}) = \frac{\text{number of trees predicting class } k}{T}$$

Soft voting gives more calibrated confidence estimates — useful when the downstream task needs probabilities, not just class labels.

---

## 6. Geometric Intuition

A single deep Decision Tree produces jagged, overfitted boundaries — it memorises the training data.

Random Forest smooths this out:
- Each tree sees different data → different boundaries
- Averaging many jagged boundaries → smoother, more robust boundary
- Increasing $T$ reduces variance — the ensemble prediction converges to the true Bayes boundary

This is the **bias-variance tradeoff** in action: individual trees have low bias but high variance; averaging reduces variance while keeping bias low.

---

## 7. Decision Boundary

![Decision Boundary](01_decision_boundary.png)

| Visual Element | Meaning |
|----------------|---------|
| Coloured regions | Predicted class regions — smooth ensemble boundary |
| White contour lines | Decision boundaries |
| Coloured dots | Training samples per class |

Compared to a single tree, the forest boundary is smoother and generalises better to unseen data.

---

## 8. Single Tree vs Forest

![Single Tree vs Forest](02_single_vs_forest.png)

Direct comparison of a single Decision Tree vs Random Forest on the same data:

- **Single tree** — jagged, overfit boundary that closely follows training noise
- **Random Forest** — smoother boundary that captures the true class structure

The forest's boundary is the average of many imperfect trees — and their combined wisdom beats any individual.

---

## 9. Build Pipeline

![Build Pipeline](03_build_pipeline.png)

Five-step training loop — once per tree:

| Step | Operation | Detail |
|------|-----------|--------|
| ① | Bootstrap sample | Draw $n$ rows with replacement |
| ② | Random features | Sample $k = \lfloor\sqrt{p}\rfloor$ features at each split |
| ③ | Build tree | Fit `DecisionTree` on bootstrap sample |
| ④ | Store tree | Append to `self.trees_` |
| ⑤ | Majority vote | At predict time — argmax over all tree votes |

---

## 10. Predicted vs Actual

![Predicted vs Actual and Model Summary](04_predicted_vs_actual.png)

**Left panel:** correct predictions shown as coloured dots. Red ✗ markers show misclassified test samples.

**Right panel:** full model summary — n_estimators, max_depth, max_features, classes, n_features, accuracy.

---

## 11. Effect of n_estimators

![Effect of n_estimators](05_n_estimators_effect.png)

Accuracy plotted against number of trees:

- Few trees → high variance — accuracy fluctuates
- More trees → accuracy stabilises — law of large numbers
- After ~50–100 trees, adding more trees gives diminishing returns
- More trees never hurt accuracy — they only cost compute time

---

## 12. Feature Importance

![Feature Importance](06_feature_importance.png)

Feature importance = average Gini reduction across all splits in all trees for that feature.

Features used in many high-gain splits → high importance.
Features rarely used or used in low-gain splits → low importance.

Useful for:
- Understanding which features drive predictions
- Feature selection — removing low-importance features
- Debugging — unexpected importance rankings reveal data issues

---

## 13. Confusion Matrix & Metrics

![Confusion Matrix and Per-Class Metrics](07_confusion_matrix.png)

**Left — Confusion Matrix:** rows are true classes, columns are predicted.

**Right — Per-Class Metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | $TP / (TP + FP)$ | Of all predicted as class $k$, how many were actually $k$ |
| Recall | $TP / (TP + FN)$ | Of all true class $k$, how many were correctly identified |
| F1 | $2 \cdot P \cdot R / (P + R)$ | Harmonic mean of precision and recall |

---

## 14. Usage

### Basic fit and predict

```python
import numpy as np
from RandomForestClassifier import RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy   : {model.score(X_test, y_test):.4f}")
print(f"Classes    : {model.classes_}")
print(f"n_trees    : {len(model.trees_)}")
print(model)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
print(f"Proba[0]   : {y_proba[0]}")
```

### Comparing n_estimators

```python
for n in [1, 5, 10, 50, 100]:
    m = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    m.fit(X_train, y_train)
    print(f"n_estimators={n:4d}  Acc={m.score(X_test, y_test):.4f}")
```

### Comparing max_features

```python
for mf in ['sqrt', 'log2', 2, None]:
    m = RandomForestClassifier(n_estimators=50, max_features=mf, random_state=42)
    m.fit(X_train, y_train)
    print(f"max_features={str(mf):6s}  Acc={m.score(X_test, y_test):.4f}")
```

### Single tree vs forest

```python
from DecisionTreeClassifier import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

print(f"Single Tree Acc : {dt.score(X_test, y_test):.4f}")
print(f"Random Forest   : {rf.score(X_test, y_test):.4f}")
```

---

## 15. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **No feature scaling needed** — tree splits are threshold-based | — |
| 2 | **n_estimators** — more trees = more stable, never hurts | Plot accuracy vs n_estimators |
| 3 | **max_depth** — shallow trees reduce overfitting per tree | Cross-validate |
| 4 | **max_features** — `'sqrt'` is the standard default for classification | Try `'log2'` on high-dim data |

> **No feature scaling required** — Random Forest inherits this from Decision Trees. Splits are based on thresholds so feature magnitude doesn't matter.

> **Random state matters** — set `random_state` for reproducible results. Different seeds give slightly different forests but similar accuracy.

---

## Decision Tree vs Random Forest vs SVC

| Criterion | Decision Tree | Random Forest | SVC |
|-----------|--------------|--------------|-----|
| Boundary shape | Axis-aligned rectangles | Smooth ensemble | Max-margin (kernel) |
| Overfitting | High (deep trees) | Low (ensemble averaging) | Low (margin) |
| Feature scaling | Not needed | Not needed | Required |
| Interpretability | Very high | Medium (feature importance) | Low |
| Training speed | Fast | Slower (T trees) | Slow (SMO) |
| Handles noise | Poorly | Well | Well |
| Probabilistic output | No | Yes — vote fraction | No |

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for plots only
sklearn              # optional — for datasets only
```

---

## License

MIT
