import numpy as np


class LassoRegressor:
    """
    Linear Regression with L1 (Lasso) regularisation.
    Supports two solvers: 'cd' (Coordinate Descent) and 'pgd' (Proximal Gradient Descent).

    Parameters
    ----------
    alpha         : float, default=0.1   — regularisation strength (lambda)
    solver        : str,   default='cd'  — 'cd' or 'pgd'
    learning_rate : float, default=0.01  — step size (pgd only)
    epochs        : int,   default=1000  — number of iterations (pgd only)
    max_iter      : int,   default=1000  — max CD passes over all features (cd only)
    tol           : float, default=1e-4  — convergence tolerance (cd only)
    fit_intercept : bool,  default=True  — whether to fit an un-penalised bias term

    Attributes
    ----------
    coef_         : ndarray (n_features,) — w
    intercept_    : float                 — b
    loss_history_ : list                  — Lasso loss per iteration/epoch
    n_iter_       : int                   — actual iterations run
    """

    def __init__(self, alpha=0.1, solver="cd", learning_rate=0.01,
                 epochs=1000, max_iter=1000, tol=1e-4, fit_intercept=True):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if solver not in ("cd", "pgd"):
            raise ValueError(f"solver must be 'cd' or 'pgd', got {solver!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.alpha         = alpha
        self.solver        = solver
        self.lr            = learning_rate
        self.epochs        = epochs
        self.max_iter      = max_iter
        self.tol           = tol
        self.fit_intercept = fit_intercept

        self.coef_         = None   # w
        self.intercept_    = None   # b
        self.loss_history_ = None   # Lasso loss per iteration/epoch
        self.n_iter_       = 0      # actual iterations run

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_train and y_train sample count mismatch: "
                             f"{X.shape[0]} vs {y.shape[0]}")

        if self.solver == "cd":
            self._fit_cd(X, y)
        else:
            self._fit_pgd(X, y)

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        return X @ self.coef_ + self.intercept_   # ŷ = X·w + b

    def score(self, X_test, y_test):
        """R² score — how well the model explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)   # R² = 1 - SS_res/SS_tot

    def __repr__(self):
        if self.coef_ is None:
            return (f"LassoRegressor(alpha={self.alpha}, solver={self.solver!r}, "
                    f"learning_rate={self.lr}, epochs={self.epochs}, "
                    f"max_iter={self.max_iter}, tol={self.tol})")
        return (f"LassoRegressor(\n"
                f"  coef_={self.coef_},\n"
                f"  intercept_={self.intercept_:.4f},\n"
                f"  alpha={self.alpha}, solver={self.solver!r}, n_iter_={self.n_iter_}\n"
                f")")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _soft_threshold(self, z, threshold):
        # S(z, t) = sign(z) * max(|z| - t, 0) — sets small values to exactly zero
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    def _lasso_loss(self, X, y):
        # MSE + (alpha/m)*||w||_1
        m = X.shape[0]
        residuals = X @ self.coef_ + self.intercept_ - y
        return float(np.mean(residuals ** 2)) + float((self.alpha / m) * np.sum(np.abs(self.coef_)))

    # ── solvers ───────────────────────────────────────────────────────────────

    def _fit_cd(self, X, y):
        """
        Coordinate Descent — updates one weight at a time via soft-threshold.
        w_j* = S(rho_j, alpha/m) / z_j
            rho_j = (1/m) * x_j^T * r_j   (partial correlation)
            z_j   = (1/m) * ||x_j||^2     (column normaliser)
        Stops early when max|delta_w| < tol.
        """
        m, n = X.shape

        self.coef_         = np.zeros(n, dtype=np.float64)  # w
        self.intercept_    = 0.0                             # b
        self.loss_history_ = []

        z = np.sum(X ** 2, axis=0) / m                      # column norms squared

        for iteration in range(self.max_iter):
            w_old = self.coef_.copy()

            if self.fit_intercept:
                self.intercept_ = float(np.mean(y - X @ self.coef_))

            for j in range(n):
                # residual excluding feature j
                r_j   = y - X @ self.coef_ - self.intercept_ + X[:, j] * self.coef_[j]
                rho_j = float(X[:, j] @ r_j) / m

                if z[j] < 1e-10:
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(
                        np.array([rho_j]), self.alpha / m
                    )[0] / z[j]

            self.loss_history_.append(self._lasso_loss(X, y))
            self.n_iter_ = iteration + 1

            if np.max(np.abs(self.coef_ - w_old)) < self.tol:
                break

    def _fit_pgd(self, X, y):
        """
        Proximal Gradient Descent — gradient step on MSE then soft-threshold for L1.
        Step 1: w_half = w - lr * (1/m) * X^T(Xw + b - y)
        Step 2: w      = S(w_half, lr * alpha / m)
        """
        m, n = X.shape

        self.coef_         = np.zeros(n, dtype=np.float64)  # w
        self.intercept_    = 0.0                             # b
        self.loss_history_ = []

        threshold = self.lr * self.alpha / m

        for epoch in range(self.epochs):

            y_hat  = X @ self.coef_ + self.intercept_       # ŷ = X·w + b
            error  = y_hat - y                              # residuals

            # gradient step on smooth MSE part
            grad_w = (X.T @ error) / m
            w_half = self.coef_ - self.lr * grad_w

            # proximal step — soft-threshold handles L1
            self.coef_ = self._soft_threshold(w_half, threshold)

            if self.fit_intercept:
                self.intercept_ -= self.lr * error.mean()

            self.loss_history_.append(self._lasso_loss(X, y))

        self.n_iter_ = self.epochs

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before using predict() or score().")