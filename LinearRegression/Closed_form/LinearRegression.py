import numpy as np


class LinearRegression:
    """
    Linear Regression using the Normal Equation.
    Falls back to SVD Pseudoinverse if XᵀX is singular.

    Attributes
    ----------
    coef_ : ndarray (n_features,) — w
    intercept_ : float            — b

    """

    def __init__(self):
        self.coef_ = None      # w
        self.intercept_ = None # b

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        # Prepend 1s column so b is solved alongside w
        X = np.insert(X, 0, 1, axis=1)

        try:
            # Normal Equation: [b, w] = (XᵀX)⁻¹ Xᵀy
            weights = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            # Fallback: SVD Pseudoinverse when XᵀX is singular
            weights = np.linalg.pinv(X) @ y

        self.intercept_ = weights[0]  # b
        self.coef_ = weights[1:]      # w

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,)
        """
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
        return "LinearRegression()"