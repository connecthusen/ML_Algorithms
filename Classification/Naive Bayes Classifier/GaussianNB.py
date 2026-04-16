import numpy as np


class GaussianNB:
    """
    Parameters
    ----------
    var_smoothing : float – Fraction of the largest per-feature variance
                            added to all class variances for numerical
                            stability (default 1e-9).
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing  = var_smoothing
        self.classes_       = None
        self.theta_         = None   # (n_classes, n_features)  – per-class feature means
        self.var_           = None   # (n_classes, n_features)  – per-class feature variances
        self.class_prior_   = None   # (n_classes,)             – prior P(C)


    #  Helpers                                                             

    def _log_likelihood(self, class_idx, X):
        """

        log p(X | C) = -½ · Σ [ log(2π·σ²) + (x − μ)² / σ² ]
        """
        mean = self.theta_[class_idx]   # (n_features,)
        var  = self.var_[class_idx]     # (n_features,)

        return -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + (X - mean) ** 2 / var,
            axis=1,
        )

   
    #  Public API                                                          

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        n_samples, n_features = X_train.shape
        self.classes_         = np.unique(y_train)
        n_classes             = len(self.classes_)

        self.theta_       = np.zeros((n_classes, n_features))
        self.var_         = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X_train[y_train == c]

            self.theta_[idx]       = np.mean(X_c, axis=0)
            self.var_[idx]         = np.var(X_c,  axis=0)
            self.class_prior_[idx] = X_c.shape[0] / n_samples

        # Variance smoothing – add ε · max_global_var to every class variance
        # (mirrors sklearn's GaussianNB behaviour)
        epsilon    = self.var_smoothing * np.var(X_train, axis=0).max()
        self.var_ += epsilon

        return self

    def predict_joint_log_proba(self, X_test):
        """
        Return the joint log-probability  log P(C) + log p(X | C)
        for every sample and every class.

        Returns
        -------
        joint_log_prob : ndarray of shape (n_samples, n_classes)
        """
        X_test    = np.asarray(X_test)
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)

        joint_log_prob = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            log_prior                  = np.log(self.class_prior_[idx])
            joint_log_prob[:, idx]     = log_prior + self._log_likelihood(idx, X_test)

        return joint_log_prob

    def predict(self, X_test):
        joint_log_prob = self.predict_joint_log_proba(np.asarray(X_test))
        return self.classes_[np.argmax(joint_log_prob, axis=1)]

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == np.asarray(y_test))