import numpy as np


class RidgeRegressor:
    """
    Linear Regression with L2 (Ridge) regularisation.
    Supports two solvers: 'closed' (Normal Equation) and 'gd' (Gradient Descent).

    Parameters
    ----------
    alpha         : float, default=0.1   — regularisation strength (lambda)
    solver        : str,   default='closed' — 'closed' or 'gd'
    learning_rate : float, default=0.01  — step size (gd only)
    epochs        : int,   default=1000  — number of iterations (gd only)
    fit_intercept : bool,  default=True  — whether to fit an un-penalised bias term

    Attributes
    ----------
    coef_         : ndarray (n_features,) — w
    intercept_    : float                 — b
    loss_history_ : list or None          — MSE per epoch (gd only), None for closed
    """

    def __init__(self, alpha=0.1, solver="closed",
                 learning_rate=0.01, epochs=1000, fit_intercept=True):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if solver not in ("gd", "closed"):
            raise ValueError(f"solver must be 'gd' or 'closed', got {solver!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        self.alpha         = alpha
        self.solver        = solver
        self.lr            = learning_rate   # stored as lr — matches original
        self.epochs        = epochs
        self.fit_intercept = fit_intercept

        self.coef_         = None   # w
        self.intercept_    = None   # b
        self.loss_history_ = None   # MSE per epoch (gd only)

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

        if self.solver == "closed":
            self._fit_closed(X, y)
        else:
            self._fit_gd(X, y)

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
            return (f"RidgeRegressor(alpha={self.alpha}, solver={self.solver!r}, "
                    f"learning_rate={self.lr}, epochs={self.epochs}, "
                    f"fit_intercept={self.fit_intercept})")
        return (f"RidgeRegressor(\n"
                f"  coef_={self.coef_},\n"
                f"  intercept_={self.intercept_:.4f},\n"
                f"  alpha={self.alpha}, solver={self.solver!r}\n"
                f")")

    # solvers

    def _fit_closed(self, X, y):
        """
        Ridge Normal Equation: w* = (XᵀX + alpha·I)⁻¹ Xᵀy
        Bias folded in via prepended 1s column.
        (0,0) entry of I zeroed so bias is NOT penalised.
        """
        if self.fit_intercept:
            ones  = np.ones((X.shape[0], 1), dtype=np.float64)
            X_aug = np.hstack([ones, X])                    # (m, p+1)

            I       = np.eye(X_aug.shape[1], dtype=np.float64)
            I[0, 0] = 0.0                                   # do not penalise intercept

            A       = X_aug.T @ X_aug + self.alpha * I
            weights = np.linalg.solve(A, X_aug.T @ y)

            self.intercept_ = float(weights[0])
            self.coef_      = weights[1:]
        else:
            I               = np.eye(X.shape[1], dtype=np.float64)
            A               = X.T @ X + self.alpha * I
            self.coef_      = np.linalg.solve(A, X.T @ y)
            self.intercept_ = 0.0

        self.loss_history_ = None

    def _fit_gd(self, X, y):
        """
        Batch Gradient Descent with Ridge penalty.

        Gradients:
            dj_dw = (1/m) * Xᵀ(Xw + b - y)  +  (alpha/m) * w
            dj_db = (1/m) * sum(Xw + b - y)       ← bias NOT penalised
        """
        m, n = X.shape

        self.coef_         = np.zeros(n, dtype=np.float64)  # w
        self.intercept_    = 0.0                             # b
        self.loss_history_ = []

        for _ in range(self.epochs):

            y_hat = X @ self.coef_ + self.intercept_        # ŷ = X·w + b

            error = y_hat - y                               # residuals

            # gradients — Ridge penalty added to weights only
            dj_dw = (X.T @ error + self.alpha * self.coef_) / m
            dj_db = np.sum(error) / m

            # simultaneous update
            self.coef_ -= self.lr * dj_dw
            if self.fit_intercept:
                self.intercept_ -= self.lr * dj_db

            # track MSE each epoch
            epoch_error = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(epoch_error ** 2)))

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before using predict() or score().")