import numpy as np


class GradientDescentRegressor:
    """
    Linear Regression using Batch Gradient Descent.

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

            y_hat = (X @ self.coef_) + self.intercept_  # ŷ = X·w + b

            error = y_hat - y  # Residuals

            # Gradients
            dj_db = np.sum(error) / m
            dj_dw = (X.T @ error) / m

            # Update w and b
            self.intercept_ -= self.lr * dj_db
            self.coef_      -= self.lr * dj_dw

            self.loss_history_.append(np.mean(error ** 2))  # Track MSE

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
        return (f"GradientDescentRegressor("
                f"learning_rate={self.lr}, epochs={self.epochs})")