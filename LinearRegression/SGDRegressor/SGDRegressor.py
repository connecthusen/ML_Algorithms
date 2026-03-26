import numpy as np


class SGDRegressor:
    """
    Linear Regression using Stochastic Gradient Descent (SGD).
    Matches sklearn behaviour — shuffles data each epoch,
    updates w and b once per sample.

    Parameters
    ----------
    learning_rate : float, default=0.01
    epochs : int, default=1000

    Attributes
    ----------
    coef_ : ndarray (n_features,) — w
    intercept_ : float            — b

    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None        # w
        self.intercept_ = None   # b
        self.loss_history_ = []  # MSE per epoch

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        m, n = X.shape

        # Initialize w and b
        self.coef_ = np.zeros(n)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):

            # Shuffle dataset each epoch 
            indices = np.random.permutation(m)

            for idx in indices:
                xi, yi = X[idx], y[idx]

                y_hat = (xi @ self.coef_) + self.intercept_  # ŷ = x·w + b

                error = y_hat - yi  # Residual (single sample)

                # Gradients — single sample, no division needed
                dj_db = error
                dj_dw = xi * error

                # Update w and b
                self.intercept_ -= self.lr * dj_db
                self.coef_      -= self.lr * dj_dw

            # Track MSE over full dataset per epoch
            epoch_loss = np.mean((X @ self.coef_ + self.intercept_ - y) ** 2)
            self.loss_history_.append(epoch_loss)

        return self

    def predict(self, X_test):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        return (X @ self.coef_) + self.intercept_  # ŷ = X·w + b

    def score(self, X_test, y_test):
        # R² — how well model explains variance in y
        y = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)  # R² = 1 - SS_res/SS_tot

    def __repr__(self):
        return (f"SGDRegressor("
                f"learning_rate={self.lr}, epochs={self.epochs})")
