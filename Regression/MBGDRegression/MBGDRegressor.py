import numpy as np


class MBGDRegressor:
    """
    Linear Regression using Mini-Batch Gradient Descent.

    Splits the training set into small batches each epoch and updates
    parameters once per batch, balancing the stability of Batch GD with
    the speed of Stochastic GD.

    Parameters
    ----------
    batch_size : int, default=32
        Number of samples per mini-batch.  The last batch in an epoch may
        be smaller if ``m % batch_size != 0``.
    learning_rate : float, default=0.01
        Step size applied to every parameter update.
    epochs : int, default=1000
        Number of full passes over the training data.
    random_state : int or None, default=None
        Seed for the shuffle RNG.  Pass an integer for reproducible runs.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weight vector **w** after training.
    intercept_ : float
        Fitted bias term **b** after training.
    loss_history_ : list of float
        MSE recorded at the end of every epoch (useful for convergence plots).

    
    """

    def __init__(self,batch_size: int = 32,learning_rate: float = 0.01,epochs: int = 1000,random_state: int | None = None,) -> None:
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.loss_history_: list[float] = []


    def fit(self, X_train, y_train) -> "MBGDRegressor":
        """
        Fit the model to training data.

        Parameters
        ----------
        X_train : array-like of shape (m, n_features)
        y_train : array-like of shape (m,) or (m, 1)

        Returns
        -------
        self : MBGDRegressor
            Fitted estimator (enables method chaining).
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()   # always (m,)

        m, n = X.shape
        rng = np.random.default_rng(self.random_state)

        # Initialise parameters
        self.coef_ = np.zeros(n)          # w  — shape (n,)
        self.intercept_ = 0.0             # b  — scalar
        self.loss_history_ = []

        for _ in range(self.epochs):
            # --- shuffle once per epoch for unbiased batch ordering ---
            indices = rng.permutation(m)
            X_shuf = X[indices]
            y_shuf = y[indices]

            # --- iterate over mini-batches ---
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size          # slice handles overflow
                X_b = X_shuf[start:end]                # (batch, n)
                y_b = y_shuf[start:end]                # (batch,)
                batch_m = X_b.shape[0]                 # true size of this batch

                # 1. Forward pass: ŷ = X·w + b
                y_hat = X_b @ self.coef_ + self.intercept_  # (batch,)

                # 2. Residuals
                error = y_hat - y_b                         # (batch,)

                # 3. Gradients (averaged over the batch)
                dj_db = np.sum(error) / batch_m
                dj_dw = (X_b.T @ error) / batch_m          # (n,)

                # 4. Parameter update
                self.intercept_ -= self.lr * dj_db
                self.coef_      -= self.lr * dj_dw

            # --- epoch-level MSE (on full, un-shuffled data) ---
            epoch_error = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(epoch_error ** 2)))

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
        return X @ self.coef_ + self.intercept_   # ŷ = X·w + b

    def score(self, X_test, y_test) -> float:
        """
        Return the coefficient of determination R² on the given test data.

        R² = 1 - SS_res / SS_tot

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
        return float(1.0 - ss_res / ss_tot)    # R² = 1 - SS_res / SS_tot

  
    def __repr__(self) -> str:
        return (
            f"MBGDRegressor("
            f"batch_size={self.batch_size}, "
            f"learning_rate={self.lr}, "
            f"epochs={self.epochs})"
        )

 
    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError(
                "This MBGDRegressor instance is not fitted yet. "
                "Call fit() before using predict() or score()."
            )