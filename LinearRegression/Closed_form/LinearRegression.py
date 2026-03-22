import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None       # w (weights) 
        self.intercept_ = None  # b (bias)

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        # Prepend 1s column for the bias term
        X = np.insert(X_train, 0, 1, axis=1)

        try:
            # 1. Normal Equation (Algebraic approach)
            # weights = (XᵀX)⁻¹ Xᵀy
            weights = np.linalg.inv(X.T @ X) @ X.T @ y_train
        except np.linalg.LinAlgError:
            # 2. SVD Pseudoinverse (Scikit-learn approach)
            # This handles cases where XᵀX is not invertible (singular)
            weights = np.linalg.pinv(X) @ y_train

        self.intercept_ = weights[0]   # b → bias
        self.coef_ = weights[1:]       # w → feature weights

    def predict(self, X_test):
        # ŷ = X·w + b
        return np.dot(X_test, self.coef_) + self.intercept_
