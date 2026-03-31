import numpy as np
from typing import Literal


class LassoRegressor:
    """
    Linear Regression with L1 (Lasso) regularisation.

    Supports two solvers selected via the ``solver`` parameter:

    ``'cd'`` — Coordinate Descent
        Minimises the Lasso objective one weight at a time using the
        soft-threshold operator.  Exploits the fact that the L1 penalty
        has a closed-form 1-D solution even though it is not globally
        differentiable.  Fast and exact; the recommended default.

    ``'pgd'`` — Proximal Gradient Descen
        Iterative solver that splits each update into a gradient step on
        the smooth MSE term followed by a proximal (soft-threshold) step
        on the L1 penalty.  Easier to reason about; useful when you want
        to monitor a smooth loss curve.

    In both cases the bias term is not penalised, matching the
    convention used by scikit-learn's ``Lasso``.

    Unlike Ridge (L2), Lasso can shrink weights to exactly zero,
    performing automatic variable selection.

    Parameters
    ----------
    alpha : float, default=0.1
        Regularisation strength (λ).  Larger values force more weights
        to exactly zero.  Must be ≥ 0.
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
        the maximum absolute change in any weight is below ``tol``.
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
    loss_history_ : list of float or None
        MSE + L1 penalty recorded at the end of every iteration/epoch.
        Populated for both solvers.
    n_iter_ : int
        Actual number of iterations run (may be less than ``max_iter``
        if ``solver='cd'`` converged early).

   
    """

    def __init__(self,
        alpha: float = 0.1,
        solver: Literal["cd", "pgd"] = "cd",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
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
        self.solver = solver
        self.lr = learning_rate
        self.epochs = epochs
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] | None = None
        self.n_iter_: int = 0

    def fit(self, X_train, y_train) -> "LassoRegressor":
        """
        Fit the model to training data.

        Parameters
        ----------
        X_train : array-like of shape (m, n_features)
        y_train : array-like of shape (m,) or (m, 1)

        Returns
        -------
        self : LassoRegressor
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
        Predict target values for X_test.

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
            f"LassoRegressor("
            f"alpha={self.alpha}, "
            f"solver={self.solver!r}, "
            f"learning_rate={self.lr}, "
            f"epochs={self.epochs}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"fit_intercept={self.fit_intercept})"
        )

    # Private: solvers 

    def _soft_threshold(self, z: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft-threshold (proximal) operator for the L1 penalty.

        For a scalar z and threshold t:
            S(z, t) = sign(z) * max(|z| - t, 0)

        Effect:
            |z| > t  →  shrinks z toward zero by t
            |z| <= t →  sets z exactly to zero  (sparse solution)

        This is the key operation that allows Lasso to produce exact zeros,
        unlike Ridge which only shrinks toward zero asymptotically.
        """
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    def _lasso_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the full Lasso objective: MSE + (alpha/m)*||w||_1."""
        m = X.shape[0]
        residuals = X @ self.coef_ + self.intercept_ - y
        mse = float(np.mean(residuals ** 2))
        l1  = float((self.alpha / m) * np.sum(np.abs(self.coef_)))
        return mse + l1

    def _fit_cd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Coordinate Descent solver with soft-threshold updates.

        For each feature j, holding all other weights fixed, the
        1-D Lasso sub-problem has the closed-form solution:

            w_j* = S(rho_j, alpha/m) / z_j

        where:
            rho_j = (1/m) * x_j^T * r_j        (partial correlation)
            r_j   = y - X_{-j} w_{-j} - b      (residual excluding feature j)
            z_j   = (1/m) * ||x_j||^2           (normalising factor)
            S     = soft-threshold operator

        The bias b is updated separately without the L1 penalty.

        Convergence is declared when max|delta_w| < tol across a full
        pass over all features.  loss_history_ is recorded each pass.
        """
        m, n = X.shape
        self.coef_         = np.zeros(n, dtype=np.float64)   # w  — shape (n,)
        self.intercept_    = 0.0                              # b  — scalar
        self.loss_history_ = []

        # Pre-compute column norms squared: z_j = (1/m) * ||x_j||^2
        z = np.sum(X ** 2, axis=0) / m                       # (n,)

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

                # Soft-threshold update
                if z[j] < 1e-10:                             # skip zero-variance feature
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(
                        np.array([rho_j]), self.alpha / m
                    )[0] / z[j]

            # Record loss and check convergence
            self.loss_history_.append(self._lasso_loss(X, y))
            self.n_iter_ = iteration + 1

            if np.max(np.abs(self.coef_ - w_old)) < self.tol:
                break   # converged early

    def _fit_pgd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Proximal Gradient Descent solver.

        Each update is a two-step operation:

        Step 1 — Gradient step on the smooth MSE term:
            w_half = w - lr * (1/m) * X^T(Xw + b - y)

        Step 2 — Proximal step on the L1 penalty (soft-threshold):
            w <- S(w_half, lr * alpha / m)

        The bias b is updated with a plain gradient step (no penalty):
            b <- b - lr * (1/m) * sum(Xw + b - y)

        This is equivalent to subgradient descent but is more stable
        because the proximal operator handles the non-smooth L1 term
        exactly rather than approximating its subgradient.
        MSE + L1 loss is recorded after every epoch.
        """
        m, n = X.shape

        # Initialise parameters
        self.coef_         = np.zeros(n, dtype=np.float64)   # w  — shape (n,)
        self.intercept_    = 0.0                              # b  — scalar
        self.loss_history_ = []

        threshold = self.lr * self.alpha / m                  # soft-threshold amount

        for epoch in range(self.epochs):
            # 1. Forward pass: y_hat = X.w + b
            y_hat = X @ self.coef_ + self.intercept_         # (m,)

            # 2. Residuals
            error = y_hat - y                                 # (m,)

            # 3. Gradient step on MSE (smooth part only)
            grad_w = (X.T @ error) / m                        # (n,)
            w_half = self.coef_ - self.lr * grad_w

            # 4. Proximal step — apply soft-threshold for L1 penalty
            self.coef_ = self._soft_threshold(w_half, threshold)

            # 5. Bias update (no regularisation)
            if self.fit_intercept:
                self.intercept_ -= self.lr * error.mean()

            # 6. Epoch-level Lasso loss (MSE + L1 penalty)
            self.loss_history_.append(self._lasso_loss(X, y))

        self.n_iter_ = self.epochs

    #  Private: guard 

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "This LassoRegressor instance is not fitted yet. "
                "Call fit() before using predict() or score()."
            )


