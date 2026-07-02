import numpy as np


class SVMClassifier:
    """
    Binary Support Vector Machine trained via Sub-Gradient Descent.

    Uses the primal hinge-loss objective — no kernel, no dual, no QP solver.
    Labels are converted internally to {-1, +1} regardless of how they're passed in.

    Parameters
    ----------
    learning_rate : float, default=0.001
    C             : float, default=1.0   — regularisation strength (higher C = less regularisation)
    epochs        : int,   default=1000

    Attributes
    ----------
    coef_         : ndarray (n_features,) — w
    intercept_    : float                 — b
    loss_history_ : list  — hinge loss recorded at every epoch
    """

    def __init__(self, learning_rate=0.001, C=1.0, epochs=1000):
        self.lr             = learning_rate
        self.C              = C
        self.epochs         = epochs
        self.coef_          = None   # w
        self.intercept_     = None   # b
        self.loss_history_  = []     # hinge loss per epoch

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,) — any binary labels, converted to {-1, +1}
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

        y = np.where(y <= 0, -1, 1)  # labels -> {-1, +1}

        m, n = X.shape

        # zero initialisation
        self.coef_         = np.zeros(n)
        self.intercept_    = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):

            indices = np.random.permutation(m)  # shuffle every epoch
            X_shuf, y_shuf = X[indices], y[indices]

            for i in range(m):
                xi, yi = X_shuf[i], y_shuf[i]

                margin = yi * (np.dot(xi, self.coef_) + self.intercept_)  # y_i(w·x_i + b)

                if margin >= 1:
                    # point is outside the margin — only the regulariser pulls w down
                    dj_dw = self.coef_
                    dj_db = 0.0
                else:
                    # point violates the margin — hinge loss contributes a gradient
                    dj_dw = self.coef_ - self.C * yi * xi
                    dj_db = -self.C * yi

                # update immediately, one sample at a time
                self.coef_      -= self.lr * dj_dw
                self.intercept_ -= self.lr * dj_db

            # log hinge loss over the full dataset once per epoch
            margins  = y * (X @ self.coef_ + self.intercept_)
            hinge    = np.maximum(0, 1 - margins)
            epoch_loss = 0.5 * np.dot(self.coef_, self.coef_) + self.C * np.mean(hinge)
            self.loss_history_.append(epoch_loss)

        return self

    def decision_function(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : raw scores (n_samples,) — signed distance from the hyperplane
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before decision_function().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        return (X @ self.coef_) + self.intercept_  # w·x + b

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — labels in {-1, +1}
        """
        scores = self.decision_function(X_test)
        return np.where(scores >= 0, 1, -1)

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        y    = np.asarray(y_test, dtype=np.float64).ravel()
        y    = np.where(y <= 0, -1, 1)
        pred = self.predict(X_test)

        return np.mean(pred == y)

    def __repr__(self):
        if self.coef_ is None:
            return (f"SVMClassifier("
                    f"learning_rate={self.lr}, C={self.C}, epochs={self.epochs})")
        return (f"SVMClassifier(\n"
                f"  coef_={self.coef_},\n"
                f"  intercept_={self.intercept_:.4f},\n"
                f"  learning_rate={self.lr}, C={self.C}, epochs={self.epochs}\n"
                f")")