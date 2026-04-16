import numpy as np


class MultinomialNB:
    """
    Parameters
    ----------
    alpha : float – Additive (Laplace/Lidstone) smoothing parameter
                    applied to every feature count to avoid zero
                    probabilities (0 for no smoothing, default 1.0).
    """

    def __init__(self, alpha=1.0):
        self.alpha              = alpha
        self.classes_           = None
        self.feature_log_prob_  = None   # (n_classes, n_features) – log P(x_i | C)
        self.class_log_prior_   = None   # (n_classes,)            – log P(C)


    #  Helpers

    def _joint_log_proba(self, X):
        """
        log P(C) + log p(X | C)  via the identity:

        log p(X | C) = X · log P(feature | C)ᵀ
        """
        return X @ self.feature_log_prob_.T + self.class_log_prior_


    #  Public API

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        n_samples, n_features = X_train.shape
        self.classes_         = np.unique(y_train)
        n_classes             = len(self.classes_)

        feature_count = np.zeros((n_classes, n_features))
        class_count   = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X_train[y_train == c]

            feature_count[idx] = np.sum(X_c, axis=0)
            class_count[idx]   = X_c.shape[0]

        # Laplace / Lidstone smoothing – add α to every feature count
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.class_log_prior_  = np.log(class_count / n_samples)

        return self

    def predict_joint_log_proba(self, X_test):
        """
        Return the joint log-probability  log P(C) + log p(X | C)
        for every sample and every class.

        Returns
        -------
        joint_log_prob : ndarray of shape (n_samples, n_classes)
        """
        return self._joint_log_proba(np.asarray(X_test))

    def predict(self, X_test):
        joint_log_prob = self.predict_joint_log_proba(np.asarray(X_test))
        return self.classes_[np.argmax(joint_log_prob, axis=1)]

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == np.asarray(y_test))