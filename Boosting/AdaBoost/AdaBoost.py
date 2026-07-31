import numpy as np


class DecisionStump:
    """A single-feature, single-threshold weak classifier."""

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.polarity  = 1      # flips which side predicts +1
        self.alpha     = None   # this stump's voting weight

    def predict(self, X):
        n_samples = X.shape[0]
        column    = X[:, self.feature]
        pred      = np.ones(n_samples)

        if self.polarity == 1:
            pred[column < self.threshold] = -1
        else:
            pred[column > self.threshold] = -1

        return pred


class AdaBoostClassifier:
    """
    Parameters
    ----------
    n_estimators : int, default=50   — number of decision stumps to boost
    random_state : int, default=None — seed for reproducibility

    Attributes
    ----------
    stumps_  : list of DecisionStump — fitted weak learners with their alpha weights
    classes_ : ndarray (2,)          — original class labels seen during fit
    """

    def __init__(self, n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.stumps_       = []
        self.classes_      = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,) — any binary labels
        """
        X     = np.asarray(X_train, dtype=np.float64)
        y_raw = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y_raw.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y_raw.shape[0]}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y_raw)
        y = np.where(y_raw == self.classes_[0], -1, 1)   # map labels to {-1, +1}

        n_samples, n_features = X.shape
        weights = np.full(n_samples, 1 / n_samples)   # every sample starts equally important
        self.stumps_ = []

        for _ in range(self.n_estimators):
            stump = self._best_stump(X, y, weights)

            predictions = stump.predict(X)
            error       = np.sum(weights[predictions != y])
            error       = np.clip(error, 1e-10, 1 - 1e-10)   # keep log() finite

            stump.alpha = 0.5 * np.log((1 - error) / error)

            weights *= np.exp(-stump.alpha * y * predictions)   # upweight misclassified samples
            weights /= weights.sum()

            self.stumps_.append(stump)

        return self

    def _best_stump(self, X, y, weights):
        """Scans every feature, threshold, and polarity for the lowest weighted error."""
        n_samples, n_features = X.shape
        best_stump = DecisionStump()
        best_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                for polarity in (1, -1):
                    pred = np.ones(n_samples)
                    if polarity == 1:
                        pred[X[:, feature] < threshold] = -1
                    else:
                        pred[X[:, feature] > threshold] = -1

                    error = np.sum(weights[pred != y])

                    if error < best_error:
                        best_error           = error
                        best_stump.feature   = feature
                        best_stump.threshold = threshold
                        best_stump.polarity  = polarity

        return best_stump

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — original class labels
        """
        if not self.stumps_:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        stump_votes = np.array([stump.alpha * stump.predict(X) for stump in self.stumps_])
        vote_sum    = np.sum(stump_votes, axis=0)

        return np.where(vote_sum >= 0, self.classes_[1], self.classes_[0])

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        y_pred = self.predict(X_test)
        return np.mean(y_pred == np.asarray(y_test))

    def __repr__(self):
        if not self.stumps_:
            return f"AdaBoostClassifier(n_estimators={self.n_estimators})"
        return (f"AdaBoostClassifier(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  classes_={self.classes_},\n"
                f"  n_stumps_fitted={len(self.stumps_)}\n"
                f")")