import numpy as np


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.

    Assumes each feature is normally distributed within a class, and that
    features are conditionally independent given the class (the "naive"
    assumption).

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Fraction of the largest per-feature variance added to every class
        variance, for numerical stability.

    Attributes
    ----------
    classes_     : ndarray (n_classes,)              — unique class labels
    theta_       : ndarray (n_classes, n_features)    — per-class feature means (μ)
    var_         : ndarray (n_classes, n_features)    — per-class feature variances (σ²)
    class_prior_ : ndarray (n_classes,)               — P(C)
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_      = None
        self.theta_        = None   # μ
        self.var_          = None   # σ²
        self.class_prior_  = None   # P(C)

    def _log_likelihood(self, class_idx, X):
        """log p(X | C) = -½ · Σ [ log(2π·σ²) + (x - μ)² / σ² ]"""
        mean = self.theta_[class_idx]
        var  = self.var_[class_idx]

        return -0.5 * np.sum(np.log(2.0 * np.pi * var) + (X - mean) ** 2 / var, axis=1)

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        self.theta_       = np.zeros((n_classes, n_features))
        self.var_         = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            self.theta_[idx]       = np.mean(X_c, axis=0)
            self.var_[idx]         = np.var(X_c, axis=0)
            self.class_prior_[idx] = X_c.shape[0] / n_samples

        # add epsilon * largest global variance to every class, avoids /0
        epsilon    = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += epsilon

        return self

    def predict_joint_log_proba(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : joint log-prob log P(C) + log p(X|C), shape (n_samples, n_classes)
        """
        if self.theta_ is None:
            raise RuntimeError("Call fit() before predict_joint_log_proba().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        joint_log_prob = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            log_prior = np.log(self.class_prior_[idx])
            joint_log_prob[:, idx] = log_prior + self._log_likelihood(idx, X)

        return joint_log_prob

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,)
        """
        joint_log_prob = self.predict_joint_log_proba(X_test)
        return self.classes_[np.argmax(joint_log_prob, axis=1)]

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        y_pred = self.predict(X_test)
        return np.mean(y_pred == np.asarray(y_test))

    def __repr__(self):
        if self.theta_ is None:
            return f"GaussianNB(var_smoothing={self.var_smoothing})"
        return (f"GaussianNB(\n"
                f"  classes_={self.classes_},\n"
                f"  class_prior_={self.class_prior_},\n"
                f"  var_smoothing={self.var_smoothing}\n"
                f")")
