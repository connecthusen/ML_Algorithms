import numpy as np

class SimpleLinearRegression:
    """
    Simple Linear Regression (one feature only).
    Fits: ŷ = w·x + b

    Attributes
    ----------
    coef_      : float — slope (w)
    intercept_ : float — intercept (b)
    """

    def __init__(self):
        self.coef_ = None       # w (slope)
        self.intercept_ = None  # b (intercept)

    def fit(self, x_train, y_train):
        """
        Input:
            x_train : (n_samples,) — single feature
            y_train : (n_samples,) — target values
        """
        x = np.asarray(x_train, dtype=np.float64).ravel()
        y = np.asarray(y_train, dtype=np.float64).ravel()

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
        self.coef_      = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        # b = ȳ - w·x̄
        self.intercept_ = y_mean - self.coef_ * x_mean

        return self

    def predict(self, x_test):
        """
        Input  : x_test (n_samples,)
        Output : y_pred (n_samples,)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")

        x = np.asarray(x_test, dtype=np.float64).ravel()
        return self.coef_ * x + self.intercept_  # ŷ = w·x + b

    def score(self, x_test, y_test):
        """R² score — how well the line fits."""
        y = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(x_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)  # R² = 1 - SS_res/SS_tot

    def __repr__(self):
        return f"SimpleLinearRegression(coef_={self.coef_}, intercept_={self.intercept_})"