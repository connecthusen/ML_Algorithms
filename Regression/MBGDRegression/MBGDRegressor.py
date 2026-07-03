import numpy as np


class MBGDRegressor:
    """
    Linear Regression using Mini-Batch Gradient Descent.

    Parameters
    ----------
    batch_size   : int,   default=32   — samples per mini-batch
    learning_rate : float, default=0.01 — step size per update
    epochs       : int,   default=1000 — full passes over training data
    random_state : int or None, default=None — seed for shuffle RNG

    Attributes
    ----------
    coef_         : ndarray (n_features,) — w
    intercept_    : float                 — b
    loss_history_ : list                  — MSE per epoch
    """

    def __init__(self, batch_size=32, learning_rate=0.01,
                 epochs=1000, random_state=None):
        self.batch_size   = batch_size
        self.lr           = learning_rate
        self.epochs       = epochs
        self.random_state = random_state

        self.coef_         = None   # w
        self.intercept_    = None   # b
        self.loss_history_ = []     # MSE per epoch

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        m, n = X.shape
        rng  = np.random.default_rng(self.random_state)

        self.coef_         = np.zeros(n)   # w
        self.intercept_    = 0.0           # b
        self.loss_history_ = []

        for _ in range(self.epochs):

            # shuffle once per epoch to remove ordering bias
            indices = rng.permutation(m)
            X_shuf  = X[indices]
            y_shuf  = y[indices]

            for start in range(0, m, self.batch_size):
                end     = start + self.batch_size          # slice handles overflow
                X_b     = X_shuf[start:end]                # (batch, n)
                y_b     = y_shuf[start:end]                # (batch,)
                batch_m = X_b.shape[0]                     # true batch size

                y_hat = X_b @ self.coef_ + self.intercept_ # ŷ = X·w + b
                error = y_hat - y_b                        # residuals

                dj_db = np.sum(error) / batch_m
                dj_dw = (X_b.T @ error) / batch_m         # (n,)

                self.intercept_ -= self.lr * dj_db
                self.coef_      -= self.lr * dj_dw

            # epoch-level MSE on full un-shuffled data
            epoch_error = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(epoch_error ** 2)))

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
            return (f"MBGDRegressor(batch_size={self.batch_size}, "
                    f"learning_rate={self.lr}, epochs={self.epochs})")
        return (f"MBGDRegressor(\n"
                f"  coef_={self.coef_},\n"
                f"  intercept_={self.intercept_:.4f},\n"
                f"  batch_size={self.batch_size}, learning_rate={self.lr}, "
                f"epochs={self.epochs}\n"
                f")")

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before using predict() or score().")