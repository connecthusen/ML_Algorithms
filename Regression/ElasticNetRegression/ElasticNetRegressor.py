import numpy as np
from typing import Literal


class ElasticNetRegressor:
    """
    Linear Regression with combined L1 + L2 (Elastic Net) regularisation.

    ``'cd'`` — Coordinate Descent
        Minimises the Elastic Net objective one weight at a time.
        The 1-D sub-problem has a closed-form solution via a modified
        soft-threshold operator that incorporates both penalties.
        Fast, exact, and the recommended default.

    ``'pgd'`` — Proximal Gradient Descent
        Splits each update into a gradient step on the smooth MSE + L2
        term, followed by a proximal (soft-threshold) step on the L1 term.
        Useful when you want epoch-by-epoch loss monitoring.

    In both cases the bias term is not penalised, matching the
    convention used by scikit-learn's ``ElasticNet``.

    Parameters
    ----------
    alpha : float, default=0.1
        Overall regularisation strength (λ).  Scales both the L1 and L2
        penalty terms together.  Must be ≥ 0.
    l1_ratio : float, default=0.5
        Mixing parameter between L1 and L2 penalties.
        Must be in [0, 1].
        ``l1_ratio=1.0`` → pure Lasso.
        ``l1_ratio=0.0`` → pure Ridge.
        ``l1_ratio=0.5`` → equal mix (default).
    solver : {'cd', 'pgd'}, default='cd'
        Which algorithm to use when calling ``fit``.
    learning_rate : float, default=0.01
        Step size applied to every gradient update.
        Used only when ``solver='pgd'``.
    epochs : int, default=1000
        Number of full passes over the training data.
        Used only when ``solver='pgd'``.
    max_iter : int, default=1000
        Maximum number of coordinate-descent passes over all features.
        Used only when ``solver='cd'``.
    tol : float, default=1e-4
        Convergence tolerance for coordinate descent.  Stops early when
        the maximum absolute change in any weight falls below ``tol``.
        Used only when ``solver='cd'``.
    fit_intercept : bool, default=True
        Whether to fit an un-penalised bias term **b**.
        Set to ``False`` to force the model through the origin.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weight vector **w** after training.
    intercept_ : float
        Fitted bias term **b** after training.
        Always ``0.0`` when ``fit_intercept=False``.
    loss_history_ : list of float
        Full Elastic Net loss recorded at the end of every iteration/epoch.
        Populated for both solvers.
    n_iter_ : int
        Actual number of iterations run (may be less than ``max_iter``
        if ``solver='cd'`` converged early).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        solver: Literal["cd", "pgd"] = "cd",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
        if solver not in ("cd", "pgd"):
            raise ValueError(f"solver must be 'cd' or 'pgd', got '{solver!r}'")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.lr = learning_rate
        self.epochs = epochs
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, X_train, y_train) -> "ElasticNetRegressor":
        """
        Fit the model to training data.

        Parameters
        ----------
        X_train : array-like of shape (m, n_features)
        y_train : array-like of shape (m,) or (m, 1)

        Returns
        -------
        self : ElasticNetRegressor
            Fitted estimator (enables method chaining).
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()   # always (m,)

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train and y_train have inconsistent sample counts: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        if self.solver == "cd":
            self._fit_cd(X, y)
        else:
            self._fit_pgd(X, y)

        return self

    def predict(self, X_test) -> np.ndarray:
        """
        Predict target values for *X_test*.

        Parameters
        ----------
        X_test : array-like of shape (m, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m,)
        """
        self._check_is_fitted()
        X = np.asarray(X_test, dtype=np.float64)
        return X @ self.coef_ + self.intercept_   # y_hat = X.w + b

    def score(self, X_test, y_test) -> float:
        """
        Return the coefficient of determination R^2 on the given test data.

        R^2 = 1 - SS_res / SS_tot

        Parameters
        ----------
        X_test : array-like of shape (m, n_features)
        y_test : array-like of shape (m,) or (m, 1)

        Returns
        -------
        r2 : float
            Best possible score is 1.0; a constant-prediction baseline
            scores 0.0.
        """
        self._check_is_fitted()
        y = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))   # R^2 = 1 - SS_res / SS_tot

    def __repr__(self) -> str:
        return (
            f"ElasticNetRegressor("
            f"alpha={self.alpha}, "
            f"l1_ratio={self.l1_ratio}, "
            f"solver={self.solver!r}, "
            f"learning_rate={self.lr}, "
            f"epochs={self.epochs}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"fit_intercept={self.fit_intercept})"
        )

    # Private: helpers 

    def _soft_threshold(self, z: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft-threshold (proximal) operator for the L1 component.

        For a scalar z and threshold t:
            S(z, t) = sign(z) * max(|z| - t, 0)

        Effect:
            |z| > t  →  shrinks z toward zero by t
            |z| <= t →  sets z exactly to zero  (sparsity)
        """
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    def _elasticnet_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the full Elastic Net objective:

            L = MSE  +  (alpha/m) * [ l1_ratio*||w||_1
                                    + (1-l1_ratio)*||w||_2^2 ]
        """
        m = X.shape[0]
        residuals = X @ self.coef_ + self.intercept_ - y
        mse  = float(np.mean(residuals ** 2))
        l1   = float(self.l1_ratio * np.sum(np.abs(self.coef_)))
        l2   = float((1.0 - self.l1_ratio) * np.sum(self.coef_ ** 2))
        return mse + (self.alpha / m) * (l1 + l2)

    # Private: solvers 

    def _fit_cd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Coordinate Descent solver for the Elastic Net.

        For each feature j, the 1-D Elastic Net sub-problem has the
        closed-form solution:

            w_j* = S(rho_j, l1_ratio * alpha / m)
                   ─────────────────────────────────────────
                   z_j  +  2 * (1 - l1_ratio) * alpha / m

        where:
            rho_j = (1/m) * x_j^T * r_j
                    partial correlation with residual excluding feature j

            r_j   = y - X_{-j} w_{-j} - b
                    residual when feature j is zeroed out

            z_j   = (1/m) * ||x_j||^2
                    column normaliser (pre-computed for efficiency)

            S     = soft-threshold operator  (L1 proximal step)

        The denominator `z_j + 2*(1-l1_ratio)*alpha/m` is the Ridge
        shrinkage applied after the L1 soft-threshold — this is what
        makes Elastic Net = Lasso update + Ridge denominator shrinkage.

        The bias b is updated without any penalty.
        Convergence is declared when max|delta_w| < tol over a full pass.
        """
        m, n = X.shape

        # Decompose alpha into L1 and L2 effective strengths
        l1_strength = self.alpha * self.l1_ratio          # soft-threshold numerator
        l2_strength = self.alpha * (1.0 - self.l1_ratio)  # ridge denominator term

        self.coef_         = np.zeros(n, dtype=np.float64)   # w  — shape (n,)
        self.intercept_    = 0.0                              # b  — scalar
        self.loss_history_ = []

        # Pre-compute column norms squared: z_j = (1/m) * ||x_j||^2
        z = np.sum(X ** 2, axis=0) / m                       # (n,)

        # Denominator: z_j + 2*(1-l1_ratio)*alpha/m  (Ridge shrinkage factor)
        ridge_denom = z + 2.0 * l2_strength / m              # (n,)

        for iteration in range(self.max_iter):
            w_old = self.coef_.copy()

            # Update bias (un-penalised) — analytic mean-residual formula
            if self.fit_intercept:
                self.intercept_ = float(np.mean(y - X @ self.coef_))

            # Cycle through each feature j
            for j in range(n):
                # Residual excluding feature j contribution
                r_j = y - X @ self.coef_ - self.intercept_ + X[:, j] * self.coef_[j]

                # Partial correlation
                rho_j = float(X[:, j] @ r_j) / m

                # ElasticNet update: soft-threshold then Ridge denominator shrinkage
                if ridge_denom[j] < 1e-10:                   # skip zero-variance feature
                    self.coef_[j] = 0.0
                else:
                    numerator      = self._soft_threshold(
                        np.array([rho_j]), l1_strength / m
                    )[0]
                    self.coef_[j]  = numerator / ridge_denom[j]

            # Record loss and check convergence
            self.loss_history_.append(self._elasticnet_loss(X, y))
            self.n_iter_ = iteration + 1

            if np.max(np.abs(self.coef_ - w_old)) < self.tol:
                break   # converged early

    def _fit_pgd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Proximal Gradient Descent solver for the Elastic Net.

        The Elastic Net objective is split into:
            - Smooth part  : MSE  +  (1-l1_ratio)*alpha/m * ||w||_2^2
            - Non-smooth   : l1_ratio * alpha/m * ||w||_1

        Each update is a two-step operation:

        Step 1 — Gradient step on the smooth MSE + L2 term:
            grad_w = (1/m) * X^T(Xw + b - y)
                     + 2*(1-l1_ratio)*alpha/m * w       <- Ridge gradient
            w_half = w - lr * grad_w

        Step 2 — Proximal step on the L1 term (soft-threshold only):
            w <- S(w_half,  lr * l1_ratio * alpha / m)

        Bias update (no regularisation):
            b <- b - lr * (1/m) * sum(Xw + b - y)

        Including the L2 gradient in Step 1 (rather than in the proximal
        step) keeps the proximal operator simple — it remains the standard
        soft-threshold used by Lasso.  The L2 component is handled exactly
        via the gradient, not approximated.
        Full Elastic Net loss is recorded after every epoch.
        """
        m, n = X.shape

        # Effective per-sample penalty strengths
        l1_threshold = self.lr * self.alpha * self.l1_ratio / m   # proximal threshold
        l2_grad_coef = 2.0 * self.alpha * (1.0 - self.l1_ratio) / m  # Ridge grad factor

        # Initialise parameters
        self.coef_         = np.zeros(n, dtype=np.float64)   # w  — shape (n,)
        self.intercept_    = 0.0                              # b  — scalar
        self.loss_history_ = []

        for epoch in range(self.epochs):
            # 1. Forward pass: y_hat = X.w + b
            y_hat = X @ self.coef_ + self.intercept_         # (m,)

            # 2. Residuals
            error = y_hat - y                                 # (m,)

            # 3. Gradient step — MSE gradient + Ridge (L2) gradient
            grad_mse   = (X.T @ error) / m                   # (n,)  MSE part
            grad_ridge = l2_grad_coef * self.coef_            # (n,)  L2 part
            w_half     = self.coef_ - self.lr * (grad_mse + grad_ridge)

            # 4. Proximal step — soft-threshold handles the L1 penalty
            self.coef_ = self._soft_threshold(w_half, l1_threshold)

            # 5. Bias update (no regularisation)
            if self.fit_intercept:
                self.intercept_ -= self.lr * error.mean()

            # 6. Epoch-level Elastic Net loss
            self.loss_history_.append(self._elasticnet_loss(X, y))

        self.n_iter_ = self.epochs

    # Private: guard 

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "This ElasticNetRegressor instance is not fitted yet. "
                "Call fit() before using predict() or score()."
            )

