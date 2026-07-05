import numpy as np


class MultinomialNB:
    """
    Multinomial Naive Bayes classifier.

    Parameters
    ----------
    alpha : float, default=1.0
        Laplace/Lidstone smoothing added to every feature count.

    Attributes
    ----------
    classes_          : ndarray (n_classes,)
    feature_log_prob_ : ndarray (n_classes, n_features) — log P(word | C)
    class_log_prior_  : ndarray (n_classes,)            — log P(C)
    """

    def __init__(self, alpha=1.0):
        self.alpha             = alpha
        self.classes_          = None
        self.feature_log_prob_ = None
        self.class_log_prior_  = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features) — raw feature counts
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

        feature_count = np.zeros((n_classes, n_features))
        class_count   = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count[idx] = np.sum(X_c, axis=0)
            class_count[idx]   = X_c.shape[0]

        smoothed_fc = feature_count + self.alpha          # Laplace smoothing
        smoothed_total = smoothed_fc.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_total)
        self.class_log_prior_  = np.log(class_count / n_samples)

        return self

    def predict_joint_log_proba(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : joint log-prob log P(C) + log p(X|C), shape (n_samples, n_classes)
        """
        if self.feature_log_prob_ is None:
            raise RuntimeError("Call fit() before predict_joint_log_proba().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        return X @ self.feature_log_prob_.T + self.class_log_prior_

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
        if self.feature_log_prob_ is None:
            return f"MultinomialNB(alpha={self.alpha})"
        return (f"MultinomialNB(\n"
                f"  classes_={self.classes_},\n"
                f"  class_log_prior_={self.class_log_prior_},\n"
                f"  alpha={self.alpha}\n"
                f")")
