import numpy as np


class ElasticNetRegressor:
    """
    Linear Regression with L1 + L2 (Elastic Net) regularisation.
    Supports two solvers: 'cd' (Coordinate Descent) and 'pgd' (Proximal Gradient Descent).

    Parameters
    ----------
    alpha         : float, default=0.1  — overall regularisation strength
    l1_ratio      : float, default=0.5  — L1 vs L2 mix; 0=Ridge, 1=Lasso
    solver        : str,   default='cd' — 'cd' or 'pgd'
    learning_rate : float, default=0.01 — step size (pgd only)
    epochs        : int,   default=1000 — number of iterations (pgd only)
    max_iter      : int,   default=1000 — max CD passes over all features (cd only)
    tol           : float, default=1e-4 — convergence tolerance (cd only)
    fit_intercept : bool,  default=True — whether to fit an un-penalised bias term

    Attributes
    ----------
    coef_         : ndarray (n_features,) — w
    intercept_    : float                 — b
    loss_history_ : list                  — Elastic Net loss per iteration/epoch
    n_iter_       : int                   — actual iterations run
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, solver="cd",
                 learning_rate=0.01, epochs=1000,
                 max_iter=1000, tol=1e-4, fit_intercept=True):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
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
        self.l1_ratio      = l1_ratio
        self.solver        = solver
        self.lr            = learning_rate
        self.epochs        = epochs
        self.max_iter      = max_iter
        self.tol           = tol
        self.fit_intercept = fit_intercept

        self.coef_         = None   # w
        self.intercept_    = None   # b
        self.loss_history_ = []     # Elastic Net loss per iteration/epoch
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
            return (f"ElasticNetRegressor(alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                    f"solver={self.solver!r}, learning_rate={self.lr}, "
                    f"epochs={self.epochs}, max_iter={self.max_iter}, tol={self.tol})")
        return (f"ElasticNetRegressor(\n"
                f"  coef_={self.coef_},\n"
                f"  intercept_={self.intercept_:.4f},\n"
                f"  alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"solver={self.solver!r}, n_iter_={self.n_iter_}\n"
                f")")

    def _soft_threshold(self, z, threshold):
        # S(z, t) = sign(z) * max(|z| - t, 0)
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    def _elasticnet_loss(self, X, y):
        # MSE + (alpha/m) * [l1_ratio*||w||_1 + (1-l1_ratio)*||w||_2^2]
        m         = X.shape[0]
        residuals = X @ self.coef_ + self.intercept_ - y
        mse       = float(np.mean(residuals ** 2))
        l1        = float(self.l1_ratio * np.sum(np.abs(self.coef_)))
        l2        = float((1.0 - self.l1_ratio) * np.sum(self.coef_ ** 2))
        return mse + (self.alpha / m) * (l1 + l2)

    def _fit_cd(self, X, y):
        """
        Coordinate Descent — updates one weight at a time.
        w_j* = S(rho_j, alpha*l1_ratio/m) / (z_j + 2*alpha*(1-l1_ratio)/m)
            rho_j = (1/m) * x_j^T * r_j   (partial correlation)
            z_j   = (1/m) * ||x_j||^2     (column normaliser)
        Stops early when max|delta_w| < tol.
        """
        m, n = X.shape

        l1_strength  = self.alpha * self.l1_ratio
        l2_strength  = self.alpha * (1.0 - self.l1_ratio)

        self.coef_         = np.zeros(n, dtype=np.float64)  # w
        self.intercept_    = 0.0                             # b
        self.loss_history_ = []

        z           = np.sum(X ** 2, axis=0) / m            # column norms squared
        ridge_denom = z + 2.0 * l2_strength / m             # Ridge shrinkage factor

        for iteration in range(self.max_iter):
            w_old = self.coef_.copy()

            if self.fit_intercept:
                self.intercept_ = float(np.mean(y - X @ self.coef_))

            for j in range(n):
                # residual excluding feature j
                r_j   = y - X @ self.coef_ - self.intercept_ + X[:, j] * self.coef_[j]
                rho_j = float(X[:, j] @ r_j) / m

                if ridge_denom[j] < 1e-10:
                    self.coef_[j] = 0.0
                else:
                    numerator     = self._soft_threshold(
                        np.array([rho_j]), l1_strength / m
                    )[0]
                    self.coef_[j] = numerator / ridge_denom[j]

            self.loss_history_.append(self._elasticnet_loss(X, y))
            self.n_iter_ = iteration + 1

            if np.max(np.abs(self.coef_ - w_old)) < self.tol:
                break

    def _fit_pgd(self, X, y):
        """
        Proximal Gradient Descent — gradient step on smooth MSE+L2, then soft-threshold for L1.
        Step 1: grad   = (1/m)*X^T(Xw+b-y) + 2*(1-l1_ratio)*alpha/m * w
                w_half = w - lr * grad
        Step 2: w      = S(w_half, lr * l1_ratio * alpha / m)
        """
        m, n = X.shape

        l1_threshold = self.lr * self.alpha * self.l1_ratio / m
        l2_grad_coef = 2.0 * self.alpha * (1.0 - self.l1_ratio) / m

        self.coef_         = np.zeros(n, dtype=np.float64)  # w
        self.intercept_    = 0.0                             # b
        self.loss_history_ = []

        for epoch in range(self.epochs):

            y_hat = X @ self.coef_ + self.intercept_        # ŷ = X·w + b
            error = y_hat - y                               # residuals

            # gradient step — MSE gradient + Ridge (L2) gradient
            grad_mse   = (X.T @ error) / m
            grad_ridge = l2_grad_coef * self.coef_
            w_half     = self.coef_ - self.lr * (grad_mse + grad_ridge)

            # proximal step — soft-threshold handles L1
            self.coef_ = self._soft_threshold(w_half, l1_threshold)

            if self.fit_intercept:
                self.intercept_ -= self.lr * error.mean()

            self.loss_history_.append(self._elasticnet_loss(X, y))

        self.n_iter_ = self.epochs

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before using predict() or score().")