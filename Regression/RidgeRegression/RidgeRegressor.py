import numpy as np
from typing import Literal
 
 
class RidgeRegressor:
    """
    Linear Regression with L2 (Ridge) regularisation.
 
    Supports two solvers selected via the ``solver`` parameter:
 
    ``'gd'`` — **Gradient Descent**
        Iterative solver.  Scales well to large datasets and many features.
        Updates parameters once per full pass over the data.
        Requires tuning ``learning_rate`` and ``epochs``.
 
    ``'closed'`` — **Closed-form Normal Equation**
        Solves  w* = (XᵀX + α·I)⁻¹ · Xᵀy  in a single matrix operation.
        Exact and fast for n_features ≲ 10 000.
        No hyperparameter tuning needed; ``learning_rate`` and ``epochs``
        are ignored.
 
    In both cases the bias term is **not** penalised, matching the
    convention used by scikit-learn's ``Ridge``.
 
    Parameters
    ----------
    alpha : float, default=0.1
        Regularisation strength (λ).  Larger values shrink the weights
        more aggressively toward zero.  Must be ≥ 0.
    solver : {'gd', 'closed'}, default='closed'
        Which algorithm to use when calling ``fit``.
    learning_rate : float, default=0.01
        Step size applied to every parameter update.
        Used only when ``solver='gd'``.
    epochs : int, default=1000
        Number of full passes over the training data.
        Used only when ``solver='gd'``.
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
        MSE recorded at the end of every epoch (useful for convergence
        plots).  ``None`` when ``solver='closed'``.
    """
 
    def __init__(self,alpha: float = 0.1,solver: Literal["gd", "closed"] = "closed",learning_rate: float = 0.01,                         epochs: int = 1000,fit_intercept: bool = True,) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if solver not in ("gd", "closed"):
            raise ValueError(f"solver must be 'gd' or 'closed', got '{solver!r}'")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
 
        self.alpha = alpha
        self.solver = solver
        self.lr = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
 
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] | None = None
 
    def fit(self, X_train, y_train) -> "RidgeRegressor":
        """
        Fit the model to training data.
 
        Parameters
        ----------
        X_train : array-like of shape (m, n_features)
        y_train : array-like of shape (m,) or (m, 1)
 
        Returns
        -------
        self : RidgeRegressor
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
 
        if self.solver == "closed":
            self._fit_closed(X, y)
        else:
            self._fit_gd(X, y)
 
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
            f"RidgeRegressor("
            f"alpha={self.alpha}, "
            f"solver={self.solver!r}, "
            f"learning_rate={self.lr}, "
            f"epochs={self.epochs}, "
            f"fit_intercept={self.fit_intercept})"
        )
 
    # Private: solvers 
 
    def _fit_closed(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Closed-form (Normal Equation) solver.
 
        Solves the regularised system exactly:
            w* = (X^T X + alpha * I_p)^-1 * X^T y
 
        When fit_intercept=True a column of ones is prepended so the bias
        is folded into the weight vector.  The (0, 0) entry of the identity
        is zeroed so the bias is NOT penalised.
 
        Time complexity: O(m*p^2 + p^3) — dominated by the matrix solve.
        loss_history_ is set to None (no iterations to record).
        """
        if self.fit_intercept:
            # Prepend a bias column of ones -> (m, p+1)
            ones  = np.ones((X.shape[0], 1), dtype=np.float64)
            X_aug = np.hstack([ones, X])
 
            I = np.eye(X_aug.shape[1], dtype=np.float64)
            I[0, 0] = 0.0                                  # do NOT penalise intercept
 
            # Solve (X^T X + alpha * I) w = X^T y
            A       = X_aug.T @ X_aug + self.alpha * I
            weights = np.linalg.solve(A, X_aug.T @ y)
 
            self.intercept_ = float(weights[0])
            self.coef_      = weights[1:]
        else:
            I = np.eye(X.shape[1], dtype=np.float64)
            A = X.T @ X + self.alpha * I
            self.coef_      = np.linalg.solve(A, X.T @ y)
            self.intercept_ = 0.0
 
        self.loss_history_ = None
 
    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Full-batch Gradient Descent solver with Ridge regularisation.
 
        Ridge objective (mean form):
            L(w, b) = (1/m)*||Xw + b - y||^2 + (alpha/m)*||w||^2
 
        Gradients:
            dL/dw = (1/m) * X^T(Xw + b - y)  +  (alpha/m) * w
            dL/db = (1/m) * sum(Xw + b - y)        <- bias NOT penalised
 
        Update rules:
            w <- w - lr * dL/dw
            b <- b - lr * dL/db
 
        Dividing alpha by m keeps the penalty on the same scale as the MSE
        gradient so the effective regularisation is dataset-size invariant.
        MSE on the full training set is recorded after every epoch.
        """
        m, n = X.shape
 
        # Initialise parameters
        self.coef_         = np.zeros(n, dtype=np.float64)  # w  — shape (n,)
        self.intercept_    = 0.0                             # b  — scalar
        self.loss_history_ = []
 
        for _ in range(self.epochs):
            # 1. Forward pass: y_hat = X.w + b
            y_hat = X @ self.coef_ + self.intercept_        # (m,)
 
            # 2. Residuals
            error = y_hat - y                               # (m,)
 
            # 3. Gradients (Ridge penalty added to weights only)
            dj_dw = (X.T @ error + self.alpha * self.coef_) / m   # (n,)
            dj_db = np.sum(error) / m                              # scalar
 
            # 4. Parameter updates
            self.coef_ -= self.lr * dj_dw
            if self.fit_intercept:
                self.intercept_ -= self.lr * dj_db
 
            # 5. Epoch-level MSE (on full, un-shuffled data)
            epoch_error = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(epoch_error ** 2)))
 
    #  Private: guard 
 
    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "This RidgeRegressor instance is not fitted yet. "
                "Call fit() before using predict() or score()."
            )
 
 