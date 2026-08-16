# AdaBoost — Adaptive Boosting

> A clean, **NumPy-only** implementation of AdaBoost for binary classification.
> Boosts a sequence of **weak learners** — decision stumps by default, or shallow trees with **gini or entropy** splits — into one strong classifier.
> Each round reweights the training data to focus on whatever the ensemble has gotten wrong so far.

---

## Table of Contents

1. [What is AdaBoost?](#1-what-is-adaboost)
2. [The Model](#2-the-model)
3. [The Weak Learners](#3-the-weak-learners)
4. [Sample Weights & Alpha](#4-sample-weights--alpha)
5. [Single Stump vs Boosted Ensemble](#5-single-stump-vs-boosted-ensemble)
6. [Decision Boundary](#6-decision-boundary)
7. [Training Pipeline](#7-training-pipeline)
8. [Test Predictions & Model Summary](#8-test-predictions--model-summary)
9. [Effect of n_estimators](#9-effect-of-n_estimators)
10. [Error & Alpha across Rounds](#10-error--alpha-across-rounds)
11. [Confusion Matrix](#11-confusion-matrix)
12. [Gini vs Entropy](#12-gini-vs-entropy)
13. [Usage](#13-usage)
14. [Assumptions](#14-assumptions)
15. [Pros & Cons vs Random Forest & a Single Decision Tree](#15-pros--cons-vs-random-forest--a-single-decision-tree)

---

## 1. What is AdaBoost?

AdaBoost ("Adaptive Boosting") builds a strong classifier out of many weak ones, trained **sequentially** rather than independently. Each new weak learner is trained on a re-weighted version of the data — samples the ensemble has been getting wrong get more weight, so the next learner is forced to pay attention to them.

Unlike a Random Forest, where every tree is trained independently and in parallel, AdaBoost's rounds depend on each other — round $t$'s sample weights are a direct consequence of how rounds $1 \ldots t{-}1$ performed.

| Symbol | Name | Meaning |
|--------|------|---------|
| $h_t(x)$ | Weak learner $t$ | One round's classifier, output in $\{-1, +1\}$ |
| $\alpha_t$ | Alpha | How much weight round $t$'s vote gets in the final prediction |
| $w_i$ | Sample weight | How much round $t{+}1$ should focus on sample $i$ |
| $\epsilon_t$ | Weighted error | Round $t$'s error rate, computed against the current weights |
| $T$ | `n_estimators` | Total number of boosting rounds |

---

## 2. The Model

The final prediction is a **weighted majority vote** across every round's weak learner:

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \, h_t(x)\right)$$

Learners that performed well (low weighted error) get a large $\alpha_t$ and dominate the vote; learners barely better than random guessing get a vote close to zero.

---

## 3. The Weak Learners

Two weak learner types are supported, selected via `weak_learner`:

**`'stump'`** (default) — a depth-1 decision tree: one feature, one threshold, one split. Classic AdaBoost. Fast, and surprisingly effective once boosted.

**`'tree'`** — a shallow decision tree (`weak_learner_depth`, default 2), split using either:
- `criterion='gini'` — $1 - \sum_c p_c^2$
- `criterion='entropy'` — $-\sum_c p_c \log_2 p_c$

The tree learner respects sample weights via a **weighted bootstrap**: rows are resampled proportional to their current weight before an ordinary (unweighted) tree is fit on the resample — a common trick that lets a plain tree-building routine behave like a weighted one.

---

## 4. Sample Weights & Alpha

At every round:

$$\epsilon_t = \sum_{i:\, h_t(x_i) \neq y_i} w_i$$

$$\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$$

$$w_i \leftarrow w_i \cdot e^{-\alpha_t y_i h_t(x_i)}, \quad \text{then renormalise so } \sum_i w_i = 1$$

The weight update is the whole mechanism in one line: get sample $i$ right and its weight shrinks; get it wrong and its weight grows — exponentially, in both directions.

---

## 5. Single Stump vs Boosted Ensemble

![Single Stump vs Boosted AdaBoost](02_stump_vs_boosted.png)

**Left:** one stump can only make a single axis-aligned cut — nowhere near enough to separate two interleaving moons. **Right:** 100 stumps combined, each focused on a different region the previous rounds struggled with, approximate a much more complex curved boundary — built entirely out of straight-line pieces.

---

## 6. Decision Boundary

![AdaBoost Decision Boundary](01_decision_boundary.png)

The characteristic **staircase** shape is the signature of a boosted-stump ensemble: every individual vote is a single axis-aligned split, and the combined boundary is the sum of a hundred small corrections layered on top of each other.

---

## 7. Training Pipeline

![AdaBoost Training Pipeline](03_training_pipeline.png)

The loop that runs once per boosting round, `n_estimators` times:

| Step | Operation |
|------|-----------|
| ① | Initialise weights — every sample starts equally important |
| ② | Fit a weak learner on the currently-weighted data |
| ③ | Compute weighted error, then this round's alpha |
| ④ | Reweight samples — boost whatever this round got wrong |
| ⑤ | Store the learner, repeat for $T$ rounds |
| ⑥ | Final prediction — weighted vote across every round |

---

## 8. Test Predictions & Model Summary

![Test Predictions and Model Summary](04_predicted_vs_actual.png)

**Left panel:** test-set predictions, with misclassified points marked by a red X. **Right panel:** model configuration and headline numbers — including the alpha and error of the very last boosting round, which is usually small since later rounds focus on the hardest remaining samples.

---

## 9. Effect of n_estimators

![Effect of n_estimators on Accuracy](05_n_estimators_effect.png)

Accuracy climbs quickly over the first ~10-20 rounds, then levels off. Because each new stump specifically targets the current ensemble's mistakes, gains come fast early on and shrink as fewer hard samples remain to fix.

---

## 10. Error & Alpha across Rounds

![Error and Alpha per Round](06_alpha_and_error.png)

**Left:** each round's weighted error — note it can spike back up when a round's weak learner draws a hard, heavily-reweighted subset. **Right:** alpha tracks error inversely — as later rounds struggle more (because the "easy" samples have already been solved), alpha for those rounds shrinks accordingly.

---

## 11. Confusion Matrix

![Confusion Matrix and Metrics](07_confusion_matrix.png)

Standard binary classification breakdown — true/false positives and negatives, plus precision, recall, and F1 computed from them.

---

## 12. Gini vs Entropy

![Gini vs Entropy Split Criterion](08_gini_vs_entropy.png)

Both impurity measures produce very similar boundaries and accuracy here — they usually agree on which split is best, differing mainly in how strongly they penalise less-pure splits. Entropy's logarithmic penalty is a bit more sensitive to small class-probability differences; gini is cheaper to compute and the more common default.

---

## 13. Usage

### Basic fit and predict — stump weak learner

```python
import numpy as np
from AdaBoostClassifier import AdaBoostClassifier

X_train = np.array([[2, 2], [3, 3], [2.5, 1.5], [-2, -2], [-3, -1], [-1.5, -2.5]])
y_train = np.array([1, 1, 1, 0, 0, 0])

model = AdaBoostClassifier(n_estimators=50, weak_learner='stump', random_state=42)
model.fit(X_train, y_train)

print(model)
print(f"Alphas (first 5) : {model.alphas_[:5]}")
print(f"Errors (first 5) : {model.errors_[:5]}")

X_test = np.array([[2.8, 2.2], [-2.2, -1.8]])
print(f"Predictions : {model.predict(X_test)}")
print(f"Accuracy    : {model.score(X_train, y_train):.4f}")
```

### Tree weak learner with entropy

```python
model = AdaBoostClassifier(
    n_estimators=50,
    weak_learner='tree',
    weak_learner_depth=2,
    criterion='entropy',
    random_state=42
)
model.fit(X_train, y_train)
print(f"Accuracy : {model.score(X_train, y_train):.4f}")
```

### Comparing n_estimators

```python
for k in [1, 5, 10, 25, 50, 100]:
    m = AdaBoostClassifier(n_estimators=k, weak_learner='stump', random_state=42)
    m.fit(X_train, y_train)
    print(f"n_estimators={k:>4} -> accuracy={m.score(X_train, y_train):.4f}")
```

---

## 14. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **Binary labels only** — this implementation does not support multi-class directly | `fit()` raises `ValueError` if more than 2 classes are given |
| 2 | **Weak learners should be better than random guessing** — AdaBoost breaks down if $\epsilon_t \geq 0.5$ consistently | Watch the `errors_` list after fitting |
| 3 | **Sensitive to noisy labels / outliers** — mislabeled points get reweighted upward every round they're missed | Cap `n_estimators`, or inspect final sample weights for outliers |
| 4 | **No feature scaling needed** — both stumps and trees split on raw thresholds | — |

> **AdaBoost can overfit with too many rounds on noisy data** — since mislabeled points keep getting boosted in weight round after round, eventually forcing the ensemble to contort around them. Cross-validate `n_estimators` rather than maximising it blindly.

---

## 15. Pros & Cons vs Random Forest & a Single Decision Tree

| Criterion | **AdaBoost** | **Random Forest** | **Single Decision Tree** |
|-----------|----------------|----------------------|------------------------------|
| Training | Sequential (each round depends on the last) | Parallel (trees are independent) | One-shot |
| Base learner | Weak — stumps or shallow trees | Full-depth trees | Itself |
| Combines via | Weighted vote (alpha) | Majority vote (equal weight) | — |
| Sensitivity to noise/outliers | High — misclassified points get boosted | Low — bootstrap averaging smooths noise | High — a single deep tree memorises noise |
| Bias vs variance | Reduces bias primarily | Reduces variance primarily | High variance if deep |
| Interpretability | Moderate — can inspect each stump | Lower — many full trees | Very high — one rule path |
| sklearn equivalent | `AdaBoostClassifier` | `RandomForestClassifier` | `DecisionTreeClassifier` |

**Rule of thumb:** reach for AdaBoost when your weak learners are consistently, if weakly, better than chance and the data is reasonably clean; prefer a Random Forest when the data is noisy, since bagging's variance reduction is more robust to outliers than boosting's bias reduction.

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for plots only
scikit-learn        # optional — for the make_moons demo dataset only
```

---

## License

MIT
