# Gaussian Naive Bayes

> A clean, **NumPy-only** implementation of Gaussian Naive Bayes.
> No gradient descent, no iterative optimisation — every parameter is computed directly from the data in one pass.
> **Bayes' theorem + a per-feature Gaussian assumption + a (naive) independence assumption.**

---

## Table of Contents

1. [What is Naive Bayes?](#1-what-is-naive-bayes)
2. [Bayes' Theorem](#2-bayes-theorem)
3. [The Gaussian Likelihood](#3-the-gaussian-likelihood)
4. [The Naive Independence Assumption](#4-the-naive-independence-assumption)
5. [Training — Closed-Form, No Iteration](#5-training--closed-form-no-iteration)
6. [Why Log-Space?](#6-why-log-space)
7. [Variance Smoothing](#7-variance-smoothing)
8. [Decision Boundary](#8-decision-boundary)
9. [Training Pipeline](#9-training-pipeline)
10. [Tuning var_smoothing](#10-tuning-var_smoothing)
11. [Usage](#11-usage)
12. [Assumptions](#12-assumptions)
13. [Pros & Cons vs Logistic Regression & SVM](#13-pros--cons-vs-logistic-regression--svm)

---

## 1. What is Naive Bayes?

Naive Bayes is a probabilistic classifier built directly on Bayes' theorem. For each class, it asks: *"if I assume this data point came from this class, how likely does it look?"* — then picks whichever class makes the observed features most probable, weighted by how common that class is to begin with.

It's called **naive** because it assumes every feature is independent of every other feature, given the class. That assumption is almost never exactly true, but the classifier is often accurate anyway — and it trains in one pass over the data, with no gradient descent required.

| Symbol | Name | Meaning |
|--------|------|---------|
| $C$ | Class | The label being predicted |
| $x_j$ | Feature $j$ | One input dimension |
| $P(C)$ | Prior | How common class $C$ is, before seeing any features |
| $p(x \mid C)$ | Likelihood | How probable feature values $x$ are, under class $C$ |
| $P(C \mid x)$ | Posterior | Updated belief about the class, after seeing $x$ |
| $\mu, \sigma^2$ | Mean, variance | Gaussian parameters fit per class, per feature |

---

## 2. Bayes' Theorem

$$P(C \mid x) = \frac{P(C) \cdot p(x \mid C)}{p(x)}$$

Since $p(x)$ is the same for every class, it doesn't affect which class wins — we only need to compare the **numerator** across classes:

$$P(C \mid x) \ \propto \ P(C) \cdot p(x \mid C)$$

![Bayes Theorem: Prior x Likelihood -> Posterior](06_bayes_theorem.png)

Prior favors Class 0 (60% vs 40%), but the likelihood at $x=3.0$ favors Class 1 strongly enough to flip the posterior — that trade-off is the entire algorithm.

---

## 3. The Gaussian Likelihood

For each class $C$ and feature $x_j$, we assume $x_j \mid C \sim \mathcal{N}(\mu_{C,j}, \sigma^2_{C,j})$:

$$p(x_j \mid C) = \frac{1}{\sqrt{2\pi\sigma^2_{C,j}}} \exp\left(-\frac{(x_j - \mu_{C,j})^2}{2\sigma^2_{C,j}}\right)$$

![Gaussian Likelihood P(x|C)](01_gaussian_likelihood.png)

**Left:** each class gets its own bell curve, fit from its own training samples. **Right:** at a given feature value, whichever class's curve is tallest is the most likely explanation for that observation.

---

## 4. The Naive Independence Assumption

Assuming features are conditionally independent given the class lets the joint likelihood factor into a simple product:

$$p(x \mid C) = \prod_{j=1}^{p} p(x_j \mid C)$$

![Naive Independence Assumption](07_independence_assumption.png)

**Left:** strongly correlated features violate the assumption — Naive Bayes will double-count the same signal across both features. **Right:** near-independent features are exactly the case the model was designed for. In practice the assumption is often violated *and* the model still performs reasonably — correlated features just get some redundant weight, they don't break the math.

---

## 5. Training — Closed-Form, No Iteration

For every class $c$, using only the samples that belong to it:

$$\mu_{c,j} = \text{mean}(x_j \text{ for samples in class } c)$$

$$\sigma^2_{c,j} = \text{var}(x_j \text{ for samples in class } c)$$

$$P(c) = \frac{\text{number of samples in class } c}{\text{total samples}}$$

That's the entire training step — no learning rate, no epochs, no convergence to wait for.

---

## 6. Why Log-Space?

Multiplying many probabilities together (one per feature) shrinks the result toward zero extremely fast — with enough features, it underflows to exactly `0.0` and every class becomes indistinguishable.

![Why Log-Space - Numerical Stability](03_log_space_stability.png)

Taking the log turns the product into a sum, which stays numerically well-behaved no matter how many features there are:

$$\log p(x \mid C) = \sum_{j=1}^{p} \log p(x_j \mid C) = -\frac{1}{2}\sum_{j=1}^{p}\left[\log(2\pi\sigma_{C,j}^2) + \frac{(x_j - \mu_{C,j})^2}{\sigma_{C,j}^2}\right]$$

Comparisons between classes still work identically in log-space, since $\log$ is monotonic — whichever class had the highest probability still has the highest log-probability.

---

## 7. Variance Smoothing

If a feature has (near) zero variance within a class, $\sigma^2 \approx 0$ makes the likelihood formula divide by zero. `var_smoothing` adds a small buffer to every class's variance to prevent this:

$$\sigma^2_{c,j} \leftarrow \sigma^2_{c,j} + \varepsilon \cdot \max_j(\text{Var}(x_j) \text{ over the whole dataset})$$

![Effect of var_smoothing on Variance Estimates](04_var_smoothing.png)

**Left:** for any variance that isn't already near zero, smoothing barely changes it. **Right:** for a variance that's dangerously close to zero, smoothing rescues it into a small but usable positive number. This mirrors exactly how `sklearn.naive_bayes.GaussianNB` handles the same issue.

---

## 8. Decision Boundary

![GaussianNB Decision Boundary - Iris Dataset](05_decision_boundary.png)

Fit on the Iris dataset's petal length and width. Note the boundary is **curved**, not a straight line — unlike linear models (GD regressor, linear SVM), Naive Bayes' boundary follows wherever the Gaussian likelihoods happen to cross over, which is generally quadratic in shape.

---

## 9. Training Pipeline

![GaussianNB Training Pipeline](02_training_pipeline.png)

The four-step flow that `fit()` runs once, in a single pass:

| Step | Operation |
|------|-----------|
| ① | Entry point — cast to arrays, validate shapes |
| ② | Per-class stats — mean, variance, and prior for every class |
| ③ | Variance smoothing — add ε · max(global variance) to every class's variance |
| ④ | Store parameters — `theta_`, `var_`, `class_prior_`, `classes_` |

---

## 10. Tuning var_smoothing

![Accuracy vs var_smoothing - Iris Dataset](08_accuracy_vs_smoothing.png)

Accuracy is flat across a wide range of `var_smoothing` values — it only matters when a class-feature variance is genuinely close to zero. Too large a value (upper-right of the plot) starts inflating every variance unnecessarily and can hurt accuracy.

---

## 11. Usage

### Basic fit and predict

```python
import numpy as np
from GaussianNB import GaussianNB

X_train = np.array([[1.0, 2.1], [1.2, 1.9], [3.9, 4.2], [4.1, 3.8]])
y_train = np.array([0, 0, 1, 1])

model = GaussianNB()
model.fit(X_train, y_train)

print(f"Classes      : {model.classes_}")
print(f"Means (θ)    : {model.theta_}")
print(f"Variances    : {model.var_}")
print(f"Priors       : {model.class_prior_}")
print(model)

X_test = np.array([[1.1, 2.0], [4.0, 4.0]])
print(f"Predictions  : {model.predict(X_test)}")
print(f"Accuracy     : {model.score(X_train, y_train):.4f}")
```

### Inspecting joint log-probabilities

```python
log_proba = model.predict_joint_log_proba(X_test)
print(log_proba)   # shape (n_samples, n_classes) — higher is more likely
```

### Tuning var_smoothing

```python
for s in [1e-12, 1e-9, 1e-6, 1e-3]:
    m = GaussianNB(var_smoothing=s)
    m.fit(X_train, y_train)
    print(f"var_smoothing={s:.0e} -> accuracy={m.score(X_train, y_train):.4f}")
```

---

## 12. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **Features are Gaussian within each class** | Plot per-class feature histograms against a fitted normal curve |
| 2 | **Conditional independence** — features don't correlate given the class | Correlation matrix within each class |
| 3 | **Enough samples per class** — mean/variance estimates need enough data to be stable | Check class sizes; very small classes get noisy estimates |
| 4 | **Features are continuous** — this implementation assumes numeric, not categorical, features | Use `CategoricalNB`/`MultinomialNB`-style models instead for discrete features |

> Naive Bayes tends to be robust even when the Gaussian assumption isn't perfect — what actually hurts it most is strong feature correlation, since it double-counts the same information.

---

## 13. Pros & Cons vs Logistic Regression & SVM

| Criterion | **Gaussian NB** | **Logistic Regression** | **Linear SVM** |
|-----------|------------------|--------------------------|-----------------|
| Training | Closed-form, one pass | Iterative (gradient descent) | Iterative (sub-gradient descent) |
| Decision boundary | Quadratic (generally curved) | Straight hyperplane | Straight hyperplane |
| Output | Class probabilities | Class probabilities | Hard label (or distance to margin) |
| Independence assumption | Yes (naive) | No | No |
| Training speed | Very fast | Moderate | Moderate |
| Works well with | Small datasets, many features (e.g. text) | General-purpose | Roughly linearly-separable data |
| Feature scaling | Not required | Recommended | Strongly required |
| sklearn equivalent | `GaussianNB` | `LogisticRegression` | `SGDClassifier(loss='hinge')` |

**Rule of thumb:** reach for Naive Bayes as a fast baseline, especially with high-dimensional data or limited training samples; move to logistic regression or an SVM when feature correlation is strong or a linear boundary isn't a good enough fit.

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for plotting only
scipy >= 1.7        # optional — for the Gaussian PDF used in these visuals
scikit-learn >= 1.0 # optional — only for the Iris demo dataset
```

---

## License

MIT
