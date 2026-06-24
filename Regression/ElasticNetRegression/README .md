# Elastic Net Regression — L1 + L2 Regularised Linear Regression

> A clean, **NumPy-only** implementation of Elastic Net Regression supporting two solvers:  
> **Coordinate Descent** (CD, fast and exact) and **Proximal Gradient Descent** (PGD, iterative).  
> Elastic Net blends the L1 and L2 penalties via `l1_ratio` —  
> **combining Ridge's stability on correlated features with Lasso's ability to produce exact zeros.**

---

## Table of Contents

1. [What is Elastic Net Regression?](#1-what-is-elastic-net-regression)
2. [The Model](#2-the-model)
3. [Cost Function — Blended Penalty](#3-cost-function--blended-penalty)
4. [Deriving the Updates](#4-deriving-the-updates)
5. [Geometric Intuition](#5-geometric-intuition)
6. [Best-Fit Line & Residuals](#6-best-fit-line--residuals)
7. [Loss Surface & PGD Trajectory](#7-loss-surface--pgd-trajectory)
8. [Derivation Pipeline](#8-derivation-pipeline)
9. [Regression Diagnostics](#9-regression-diagnostics)
10. [Predicted vs Actual](#10-predicted-vs-actual)
11. [Coefficient Path vs l1_ratio](#11-coefficient-path-vs-l1_ratio)
12. [Loss Curve — CD vs PGD](#12-loss-curve--cd-vs-pgd)
13. [Usage](#13-usage)
14. [Assumptions](#14-assumptions)

---

## 1. What is Elastic Net Regression?

**Elastic Net** is a regularised linear regression that simultaneously applies both the L1 (Lasso) and L2 (Ridge) penalties, controlled by a single mixing parameter `l1_ratio`.

Given $n$ observations $(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)$, it finds the hyperplane:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b$$

| Symbol | Name | Meaning |
|--------|------|---------|
| $w_j$ | Weight | Change in $\hat{y}$ per unit increase in $x_j$ |
| $b$ | Bias / Intercept | Value of $\hat{y}$ when all $x_j = 0$ — never penalised |
| $\hat{y}$ | Prediction | Model output for a given $\mathbf{x}$ |
| $e_i = y_i - \hat{y}_i$ | Residual | Error for sample $i$ |
| $\alpha$ | Regularisation strength | Controls overall penalty magnitude |
| `l1_ratio` $\rho$ | Mixing parameter | 0 = pure Ridge, 1 = pure Lasso |

Elastic Net solves two key limitations:

- **Lasso's instability** — when features are correlated, Lasso arbitrarily picks one and zeros the rest. Elastic Net's L2 term groups correlated features and assigns them similar weights.
- **Ridge's inability to produce zeros** — Ridge never reaches exactly zero. Elastic Net's L1 term still drives irrelevant features to exact zero.

Setting `l1_ratio=1.0` recovers **pure Lasso**. Setting `l1_ratio=0.0` recovers **pure Ridge**.

---

## 2. The Model

For $n$ samples and $p$ features the prediction is identical to OLS, Ridge, and Lasso:

$$\hat{y}_i = w_1 x_{i1} + w_2 x_{i2} + \cdots + w_p x_{ip} + b$$

In matrix form:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b, \qquad \mathbf{X} \in \mathbb{R}^{n \times p},\quad \mathbf{w} \in \mathbb{R}^{p},\quad b \in \mathbb{R}$$

> The difference from all prior methods is entirely in the penalty — Elastic Net uses a **convex combination of L1 and L2 norms** controlled by `l1_ratio`. The bias $b$ is **never penalised**.

---

## 3. Cost Function — Blended Penalty

Elastic Net minimises **MSE plus a weighted blend of L1 and L2 penalties**:

$$\mathcal{L}(\mathbf{w}, b) = \underbrace{\frac{1}{n}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|^2}_{\text{MSE}} + \underbrace{\frac{\alpha}{n}\Big[\,\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2\,\Big]}_{\text{Elastic Net penalty}}$$

where $\rho$ is `l1_ratio` and $\alpha$ is the overall regularisation strength.

Key properties:

- $\rho = 1$ → pure Lasso penalty $\frac{\alpha}{n}\|\mathbf{w}\|_1$
- $\rho = 0$ → pure Ridge penalty $\frac{\alpha}{n}\|\mathbf{w}\|_2^2$
- The bias $b$ is **not penalised** in either term.
- The surface is **strictly convex** — a unique global minimum always exists.

| `l1_ratio` | Behaviour |
|-----------|-----------|
| `0.0` | Pure Ridge — no zeros, maximum stability |
| `0.1–0.3` | Ridge-dominant — few zeros |
| `0.5` | Equal mix — good default |
| `0.7–0.9` | Lasso-dominant — more zeros |
| `1.0` | Pure Lasso — maximum sparsity |

---

## 4. Deriving the Updates

### Coordinate Descent (CD) — recommended

For each feature $j$, holding all others fixed, the 1-D Elastic Net sub-problem has the closed-form solution:

$$w_j^* = \frac{S\!\left(\rho_j,\;\dfrac{\alpha\,\rho}{n}\right)}{z_j + \dfrac{2\,\alpha\,(1-\rho)}{n}}$$

where:

$$\rho_j = \frac{1}{n}\,\mathbf{x}_j^T\,\mathbf{r}_j \quad \text{(partial correlation)}, \qquad z_j = \frac{1}{n}\|\mathbf{x}_j\|^2 \quad \text{(column normaliser)}$$

$$S(z,\, t) = \text{sign}(z)\cdot\max(|z| - t,\; 0) \quad \text{(soft-threshold operator)}$$

Compared to Lasso CD, Elastic Net has one extra term in the denominator: $\dfrac{2\alpha(1-\rho)}{n}$ — the Ridge shrinkage factor. When $\rho=1$ this vanishes (pure Lasso); when $\rho=0$ only Ridge shrinkage remains.

### Proximal Gradient Descent (PGD)

The Elastic Net objective is split into smooth (MSE + L2) and non-smooth (L1) parts:

**Step 1 — Gradient step on smooth MSE + L2:**
$$\mathbf{g} = \frac{1}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} + b - \mathbf{y}) + \frac{2\alpha(1-\rho)}{n}\mathbf{w}, \qquad \mathbf{w}_{½} = \mathbf{w} - \eta\,\mathbf{g}$$

**Step 2 — Proximal step on L1 only (soft-threshold):**
$$\mathbf{w} \leftarrow S\!\left(\mathbf{w}_{½},\;\frac{\eta\,\alpha\,\rho}{n}\right)$$

**Bias update (no penalty):**
$$b \leftarrow b - \eta \cdot \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

---

## 5. Geometric Intuition

| Method | Constraint shape | Exact zeros? | Correlated features |
|--------|-----------------|-------------|-------------------|
| OLS | None | No | Unstable |
| Ridge | Sphere — smooth | No | Grouped, stable |
| Lasso | Diamond — sharp corners | Yes | Arbitrary selection |
| Elastic Net | Rounded diamond | Yes | Grouped + selected |

The L2 component smooths the L1 corners of the constraint region — solutions can still land at a corner (exact zero) but are more stable near correlated features. This is the **grouping effect**: correlated features receive similar weights instead of competing against each other.

| Scenario | Lasso | Elastic Net |
|----------|-------|-------------|
| Two correlated informative features | Picks one arbitrarily | Keeps both with similar weights |
| Many irrelevant features | Zeros them out | Zeros them out |
| $p > n$ | Selects at most $n$ | Can select more than $n$ |

---

## 6. Best-Fit Line & Residuals

![Best-Fit Line and Residuals](01_bestfit_residuals.png)

| Visual Element | Meaning |
|----------------|---------|
| Blue dots | Observed data points $(x_i,\ y_i)$ |
| Red line | Elastic Net best-fit line (`l1_ratio=0.5`) |
| Green bars | Small residuals — points close to the line |
| Pink bars | Large residuals — points far from the line |

The L1+L2 blend produces slight weight shrinkage compared to OLS, while retaining the ability to zero out irrelevant features.

---

## 7. Loss Surface & PGD Trajectory

![Elastic Net Loss Surface with PGD Trajectory](02_loss_surface.png)

The contour map shows the Elastic Net loss surface over slope $w$ and bias $b$ with `l1_ratio=0.5`.

- The surface is **convex** — one global minimum guaranteed.
- The **amber path** is the PGD trajectory from the yellow start toward the green converged minimum.
- Smooth contours reflect the L2 component; the slight asymmetry near $w=0$ reflects the L1 component.

---

## 8. Derivation Pipeline

![PGD Pipeline and Constraint Geometry](03_gradient_derivation.png)

**Left — PGD five-step loop per epoch:**

| Step | Operation | Formula |
|------|-----------|---------|
| ① | Forward pass | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ |
| ② | Residuals | $\varepsilon = \hat{\mathbf{y}} - \mathbf{y}$ |
| ③ | Smooth gradient | $\nabla_{\text{MSE}} + 2\lambda_2\mathbf{w}$ |
| ④ | Soft-threshold | $S(\mathbf{w}_{½},\; \lambda_1/n)$ |
| ⑤ | Bias update | $b \leftarrow b - \eta\bar{\varepsilon}$ |

**Right — Constraint geometry:** Elastic Net feasible regions (amber/purple curves) interpolate between the L2 ball (blue circle) and the L1 diamond (pink), producing rounded corners that can still touch the axes.

---

## 9. Regression Diagnostics

After fitting, verify the four core assumptions visually:

![Regression Diagnostics](04_diagnostics.png)

| Plot | What to look for | Assumption verified |
|------|-----------------|---------------------|
| **Residuals vs Fitted** | Random scatter around $y=0$, no curve | Linearity |
| **Normal Q-Q** | Points on the diagonal line | Normality of residuals |
| **Scale-Location** | Flat, uniform band — no funnel | Homoscedasticity |
| **Residual Histogram** | Bell-shaped, centred at 0 | Normality |

**Red flags:**
- Curve in *Residuals vs Fitted* → relationship is non-linear; try feature transformation
- Funnel shape in *Scale-Location* → variance not constant; try log($y$)
- Heavy tails in Q-Q → residuals not normal; consider robust regression

---

## 10. Predicted vs Actual

![Predicted vs Actual and Weight Comparison](05_multivariate.png)

**Left panel:** each point is one sample — actual $y$ on x-axis, predicted $\hat{y}$ on y-axis.
- Points hugging the **red dashed diagonal** = accurate predictions.

**Right panel:** learned weights compared across Lasso, Elastic Net, and Ridge at the same `alpha`. Elastic Net sits between the two extremes — some features zeroed like Lasso, remaining weights more evenly distributed like Ridge.

**Model summary:**

| Metric | Meaning |
|--------|---------|
| $R^2$ | Proportion of variance in $y$ explained by the model |
| MSE | Mean squared error — average squared residual |
| Non-zero $w$ | Features kept — rest driven to exactly zero |
| `l1_ratio` | Blend used — 0=Ridge, 1=Lasso |
| `n_iter_` | Actual iterations run |

---

## 11. Coefficient Path vs l1_ratio

![Coefficient Path vs l1_ratio](06_l1ratio_path.png)

As `l1_ratio` sweeps from 0 (pure Ridge) to 1 (pure Lasso), each coefficient traces a path showing how the L1/L2 blend affects it.

**Left (`alpha=0.1`):** mild regularisation — coefficients change gradually across the path.
**Right (`alpha=0.5`):** strong regularisation — Ridge end shows shrunk non-zero weights; Lasso end shows sparser, more aggressively zeroed weights.

| Observation | Interpretation |
|-------------|---------------|
| Coefficient flat across path | Feature robust to L1/L2 trade-off |
| Coefficient zeros only at Lasso end | Weakly informative — removed only under strong L1 |
| Coefficient zero throughout | Feature irrelevant regardless of blend |
| All paths flat | `alpha` too small — no regularisation effect yet |

---

## 12. Loss Curve — CD vs PGD

`loss_history_` stores the full Elastic Net loss at the end of every iteration or epoch for both solvers.

![Loss Curve — CD vs PGD](07_loss_curve.png)

**Left (CD):** converges in very few iterations. Check `n_iter_` after fitting — if it equals `max_iter`, increase `max_iter`.

**Right (PGD):** sharp initial drop followed by smooth flattening. Log-scale inset confirms clean monotone decay. If the curve oscillates → reduce `learning_rate`.

---

## 13. Usage

### Basic fit and predict

```python
import numpy as np
from ElasticNetRegressor import ElasticNetRegressor

X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
y_train = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

# coordinate descent — recommended default
model = ElasticNetRegressor(alpha=0.1, l1_ratio=0.5, solver='cd')
model.fit(X_train, y_train)

print(f"Intercept (b) : {model.intercept_:.4f}")
print(f"Weights   (w) : {model.coef_}")
print(f"n_iter_       : {model.n_iter_}")
print(model)

X_test = np.array([[6], [7], [8]], dtype=float)
y_test = np.array([12.0, 13.8, 16.1])
y_pred = model.predict(X_test)

print(f"Predictions   : {y_pred}")
print(f"R²            : {model.score(X_test, y_test):.4f}")
```

### PGD solver with loss curve

```python
model = ElasticNetRegressor(alpha=0.1, l1_ratio=0.5, solver='pgd',
                            learning_rate=0.05, epochs=2000)
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
plt.plot(model.loss_history_)
plt.xlabel("Epoch")
plt.ylabel("Elastic Net Loss")
plt.title("ElasticNet PGD — Loss Curve")
plt.show()
```

### Multi-feature sparse example

```python
X_multi = np.random.randn(200, 8)
true_w  = np.array([2.5, -1.8, 0.0, 3.2, 0.0, 0.0, 1.1, 0.0])
y_multi = X_multi @ true_w + np.random.randn(200)

model = ElasticNetRegressor(alpha=0.1, l1_ratio=0.5,
                            solver='cd', max_iter=2000, tol=1e-6)
model.fit(X_multi, y_multi)

print(f"R²         : {model.score(X_multi, y_multi):.4f}")
print(f"Non-zero w : {(model.coef_ != 0).sum()} / {X_multi.shape[1]}")
print(model)
```

### Sweeping l1_ratio

```python
for l1r in [0.0, 0.3, 0.5, 0.7, 1.0]:
    m = ElasticNetRegressor(alpha=0.1, l1_ratio=l1r, solver='cd')
    m.fit(X_multi, y_multi)
    print(f"l1_ratio={l1r:.1f}  |  R²={m.score(X_multi, y_multi):.4f}  "
          f"|  non-zero={(m.coef_ != 0).sum()}")
```

### Recovering pure Lasso or pure Ridge

```python
lasso_model = ElasticNetRegressor(alpha=0.1, l1_ratio=1.0, solver='cd')
ridge_model  = ElasticNetRegressor(alpha=0.1, l1_ratio=0.0, solver='cd')
```

---

## 14. Assumptions

| # | Assumption | How to check |
|---|-----------|--------------|
| 1 | **Linearity** — true relationship is $y = \mathbf{X}\mathbf{w} + b + \varepsilon$ | Residuals vs Fitted plot |
| 2 | **Zero-mean errors** — $\mathbb{E}[\varepsilon] = 0$ | Residual histogram centred at 0 |
| 3 | **Homoscedasticity** — $\text{Var}(\varepsilon_i) = \sigma^2$ constant | Scale-Location plot |
| 4 | **Independent errors** — $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ | Durbin-Watson test |
| 5 | **Normality** *(inference only)* — $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ | Normal Q-Q plot |

> **Feature scaling is required** — Elastic Net is sensitive to feature scale. Always apply `StandardScaler` before fitting; without it, features with larger magnitudes receive disproportionately large penalties.

> **Hyperparameter selection** — use cross-validation to tune both `alpha` and `l1_ratio` jointly. A 2-D grid search is standard practice.

---

## OLS vs Ridge vs Lasso vs Elastic Net

| Criterion | OLS | Ridge (L2) | Lasso (L1) | Elastic Net |
|-----------|-----|-----------|-----------|------------|
| Penalty | None | $\alpha\|\mathbf{w}\|_2^2$ | $\alpha\|\mathbf{w}\|_1$ | $\alpha[\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2]$ |
| Constraint shape | — | Sphere | Diamond | Rounded diamond |
| Exact zero weights | No | No | Yes | Yes |
| Variable selection | No | No | Yes | Yes |
| Handles correlated features | Poorly | Well | Arbitrarily | Well + selects |
| Closed-form solution | Yes | Yes | No | No |
| Unique solution | Yes (if full rank) | Always | Not always | Always |
| Key hyperparameter | — | `alpha` | `alpha` | `alpha` + `l1_ratio` |

**Rule of thumb:** use Elastic Net when you have correlated features and want both stability and sparsity. Start with `alpha=0.1, l1_ratio=0.5` and tune both via cross-validation.

---

## Dependencies

```
numpy >= 1.21
matplotlib >= 3.4   # optional — for loss curve and plots only
scipy >= 1.7        # optional — for Q-Q diagnostics
```

---

## License

MIT