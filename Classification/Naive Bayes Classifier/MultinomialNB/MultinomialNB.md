# Multinomial Naive Bayes

> A clean, **NumPy-only** implementation of Multinomial Naive Bayes.
> Built for count data — word counts, term frequencies, any feature that represents "how many times something happened."
> Same Bayes'-theorem foundation as Gaussian NB, but the likelihood is a **multinomial**, not a Gaussian.

---

## Table of Contents

1. [What is Multinomial Naive Bayes?](#1-what-is-multinomial-naive-bayes)
2. [Bayes' Theorem](#2-bayes-theorem)
3. [The Multinomial Likelihood](#3-the-multinomial-likelihood)
4. [The Naive Independence Assumption](#4-the-naive-independence-assumption)
5. [Training — Closed-Form, No Iteration](#5-training--closed-form-no-iteration)
6. [Why Log-Space?](#6-why-log-space)
7. [Laplace Smoothing](#7-laplace-smoothing)
8. [Decision Boundary](#8-decision-boundary)
9. [Training Pipeline](#9-training-pipeline)
10. [Tuning alpha](#10-tuning-alpha)
11. [Usage](#11-usage)
12. [Assumptions](#12-assumptions)
13. [Pros & Cons vs Gaussian NB & Logistic Regression](#13-pros--cons-vs-gaussian-nb--logistic-regression)

---

## 1. What is Multinomial Naive Bayes?

Multinomial NB is the version of Naive Bayes built for **count data** — the classic example is spam filtering, where each feature is "how many times word $j$ appeared in this document." Instead of assuming each feature is Gaussian (as in `GaussianNB`), it models the whole feature vector for a class as draws from a multinomial distribution over the vocabulary.

| Symbol | Name | Meaning |
|--------|------|---------|
| $C$ | Class | The label being predicted |
| $x_j$ | Feature $j$ | Count of word/token $j$ in a sample |
| $P(C)$ | Prior | How common class $C$ is |
| $P(x_j \mid C)$ | Per-feature probability | How often word $j$ shows up in class $C$'s documents |
| $\alpha$ | Smoothing parameter | Prevents zero probabilities for unseen words |

---

## 2. Bayes' Theorem

$$P(C \mid x) = \frac{P(C) \cdot p(x \mid C)}{p(x)}$$

$p(x)$ is identical across classes, so classification only needs the numerator:

$$P(C \mid x) \ \propto \ P(C) \cdot p(x \mid C)$$

![Bayes Theorem: Prior x Likelihood -> Posterior](06_bayes_theorem.png)

Sport starts with the highest prior (50%), but Tech's much stronger likelihood for this particular document flips the posterior decisively in Tech's favor.

---

## 3. The Multinomial Likelihood

Given a document with word counts $x = (x_1, \ldots, x_p)$, the multinomial likelihood under class $C$ is:

$$p(x \mid C) \ \propto \ \prod_{j=1}^{p} P(x_j \mid C)^{x_j}$$

$P(x_j \mid C)$ is estimated as the fraction of all word occurrences in class $C$'s training documents that were word $j$:

$$P(x_j \mid C) = \frac{\text{count of word } j \text{ across all class-}C\text{ documents}}{\text{total word count across all class-}C\text{ documents}}$$

![Multinomial Likelihood - Word Counts](01_multinomial_likelihood.png)

**Left:** each class has its own word-probability profile — "Python" and "Neural" dominate Tech, "Buy" and "Deal" dominate Spam. **Right:** a new document's log-likelihood under each class is just a weighted sum of these per-word log-probabilities — whichever class scores highest wins.

---

## 4. The Naive Independence Assumption

Just like Gaussian NB, this model assumes each word's count is conditionally independent of every other word's count, given the class:

$$p(x \mid C) = \prod_{j=1}^{p} p(x_j \mid C)^{x_j}$$

![Naive Independence Assumption](07_independence_assumption.png)

**Left:** strongly correlated features (words that always co-occur) violate the assumption — the model effectively counts the same signal twice. **Right:** near-independent features match the assumption cleanly. In text data, some correlation between words is unavoidable (e.g. "machine" and "learning"), but the model tends to work well in practice regardless.

---

## 5. Training — Closed-Form, No Iteration

For every class $c$:

$$P(x_j \mid c) = \frac{\left(\sum_{i \in c} x_{ij}\right) + \alpha}{\sum_{j'=1}^{p}\left[\left(\sum_{i \in c} x_{ij'}\right) + \alpha\right]}$$

$$P(c) = \frac{\text{number of samples in class } c}{\text{total samples}}$$

Every parameter comes directly from counting — no gradient descent, no epochs.

---

## 6. Why Log-Space?

Multiplying many small per-word probabilities together underflows to exactly `0.0` once the vocabulary or document length grows.

![Why Log-Space - Numerical Stability](03_log_space_stability.png)

Working in log-space turns the product into a sum, computed in one shot via:

$$\log p(x \mid C) = x \cdot \log P(\cdot \mid C)^{T}$$

— a single dot product between the count vector and the class's log-probability vector, which is exactly what `predict_joint_log_proba` computes.

---

## 7. Laplace Smoothing

Any word that never appeared in a class's training documents gets $P(x_j \mid C) = 0$ — and since the likelihood is a *product*, one zero would make the entire document's probability zero for that class, no matter how well the other words fit. `alpha` fixes this by adding a small constant count to every word before normalising:

$$P(x_j \mid c) = \frac{\text{count}(x_j, c) + \alpha}{\sum_{j'}\left[\text{count}(x_{j'}, c) + \alpha\right]}$$

![Effect of Laplace Smoothing on Zero-Count Features](04_laplace_smoothing.png)

**Left:** words with zero training counts (w3, w5) get a small non-zero probability instead of exactly zero. **Right:** larger `alpha` pulls every class toward a uniform word distribution, which lowers the training log-likelihood — `alpha` trades a perfect fit on seen data for the ability to handle unseen words gracefully.

---

## 8. Decision Boundary

![Decision Boundary - 2 word-count features](05_decision_boundary.png)

**Left:** the boundary between two classes defined by two word-count features — points with more of feature 2 relative to feature 1 fall into class 0, and vice versa. **Right:** the same boundary expressed as a smooth posterior probability surface — confidence is highest far from the boundary and approaches 0.5 right at the decision edge.

---

## 9. Training Pipeline

![MultinomialNB Training Pipeline](02_training_pipeline.png)

The full `fit()` flow, computed once in a single pass:

| Step | Operation | Stored as |
|------|-----------|-----------|
| 1–2 | Cast inputs, identify unique classes | `classes_` |
| 3 | Sum feature counts per class | `feature_count` |
| 4 | Add Laplace smoothing | `smoothed_fc` |
| 5 | Normalise into log word-probabilities | `feature_log_prob_` |
| 6 | Compute log class priors | `class_log_prior_` |

---

## 10. Tuning alpha

![Accuracy vs Laplace Smoothing Parameter alpha](08_accuracy_vs_alpha.png)

Cross-validated accuracy is essentially flat across many orders of magnitude of `alpha` — it mostly matters when the vocabulary is large relative to the training set, where zero-count words are common. The default `alpha=1.0` is a safe general-purpose choice.

---

## 11. Usage

### Basic fit and predict

```python
import numpy as np
from MultinomialNB import MultinomialNB

# rows = documents, columns = word counts
X_train = np.array([
    [5, 4, 1, 0, 6, 0],   # tech-heavy document
    [4, 3, 1, 1, 5, 0],   # tech-heavy document
    [0, 0, 4, 6, 1, 7],   # spam-heavy document
    [1, 0, 3, 7, 0, 6],   # spam-heavy document
])
y_train = np.array(["Tech", "Tech", "Spam", "Spam"])

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

print(f"Classes         : {model.classes_}")
print(f"Log word probs  : {model.feature_log_prob_}")
print(f"Log class prior : {model.class_log_prior_}")
print(model)

X_test = np.array([[3, 0, 0, 0, 1, 0]])   # mostly "Python" and "Science"
print(f"Prediction      : {model.predict(X_test)}")
print(f"Accuracy        : {model.score(X_train, y_train):.4f}")
```

### Inspecting joint log-probabilities

```python
log_proba = model.predict_joint_log_proba(X_test)
print(log_proba)   # shape (n_samples, n_classes) — higher is more likely
```

### Tuning alpha

```python
for a in [0.01, 0.1, 1.0, 5.0]:
    m = MultinomialNB(alpha=a)
    m.fit(X_train, y_train)
    print(f"alpha={a:>5} -> accuracy={m.score(X_train, y_train):.4f}")
```

---

## 12. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **Features are counts** — non-negative integers representing frequency | Not suitable for raw continuous measurements (use `GaussianNB` instead) |
| 2 | **Conditional independence** — word counts don't correlate given the class | Correlation between features within each class |
| 3 | **Bag-of-words style data** — word order doesn't matter, only frequency | True for classic text classification, not for order-sensitive tasks |
| 4 | **Vocabulary seen in training generalises** — `alpha` is tuned, not ignored | Cross-validate over a range of `alpha` |

> Word counts that are extremely skewed by document length can distort the likelihood — normalising by document length (TF or TF-IDF weighting) before fitting is a common preprocessing step in practice.

---

## 13. Pros & Cons vs Gaussian NB & Logistic Regression

| Criterion | **Multinomial NB** | **Gaussian NB** | **Logistic Regression** |
|-----------|----------------------|-------------------|---------------------------|
| Feature type | Counts / frequencies | Continuous, Gaussian-ish | Continuous or encoded categorical |
| Training | Closed-form, one pass | Closed-form, one pass | Iterative (gradient descent) |
| Typical use case | Text classification, spam filtering | Sensor/measurement data | General-purpose |
| Handles zero counts | Yes, via `alpha` smoothing | N/A (uses variance smoothing instead) | N/A |
| Decision boundary | Log-linear in counts | Quadratic (curved) | Straight hyperplane |
| Feature scaling | Not required | Not required | Recommended |
| sklearn equivalent | `MultinomialNB` | `GaussianNB` | `LogisticRegression` |

**Rule of thumb:** use Multinomial NB whenever your features are counts (words, n-grams, event tallies); reach for Gaussian NB when features are continuous measurements instead.

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for plotting only
```

---

## License

MIT
