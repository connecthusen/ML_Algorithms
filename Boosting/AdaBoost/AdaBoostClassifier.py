import numpy as np
from collections import Counter


class DecisionStump:
    """Depth-1 decision tree — the classic AdaBoost weak learner, works for any number of classes."""

    def __init__(self):
        self.feature     = None
        self.threshold   = None
        self.left_value  = None   # predicted class if x[feature] <= threshold
        self.right_value = None   # predicted class if x[feature] > threshold
        self.alpha       = None   # voting weight assigned after training

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float('inf')

        for feature in range(n_features):
            for threshold in np.unique(X[:, feature]):
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_value  = self._weighted_majority(y[left_mask],  weights[left_mask])
                right_value = self._weighted_majority(y[right_mask], weights[right_mask])

                pred  = np.where(left_mask, left_value, right_value)
                error = np.sum(weights[pred != y])

                if error < best_error:
                    best_error       = error
                    self.feature     = feature
                    self.threshold   = threshold
                    self.left_value  = left_value
                    self.right_value = right_value

    def predict(self, X):
        column = X[:, self.feature]
        return np.where(column <= self.threshold, self.left_value, self.right_value)

    def _weighted_majority(self, y_subset, w_subset):
        """Class with the highest total sample weight in this subset."""
        if len(y_subset) == 0:
            return 0
        totals = {}
        for label, weight in zip(y_subset, w_subset):
            totals[label] = totals.get(label, 0.0) + weight
        return max(totals, key=totals.get)


class WeakDecisionTree:
    """A shallow decision tree weak learner, for when stumps underfit the data."""

    def __init__(self, max_depth=2, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.alpha     = None
        self.root      = None

    def fit(self, X, y, weights):
        # a weighted bootstrap lets a normal (unweighted) tree respect sample weights
        indices   = np.random.choice(len(y), size=len(y), replace=True, p=weights)
        X_sampled = X[indices]
        y_sampled = y[indices]
        self.root = self._build(X_sampled, y_sampled, depth=0)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _build(self, X, y, depth):
        majority = Counter(y).most_common(1)[0][0]

        if len(set(y)) == 1:
            return {'leaf': True, 'value': y[0]}
        if depth >= self.max_depth or len(y) < 2:
            return {'leaf': True, 'value': majority}

        feature, threshold, gain = self._best_split(X, y)

        if gain <= 0:
            return {'leaf': True, 'value': majority}

        lm = X[:, feature] <= threshold
        rm = ~lm

        return {
            'leaf':      False,
            'feature':   feature,
            'threshold': threshold,
            'left':      self._build(X[lm], y[lm], depth + 1),
            'right':     self._build(X[rm], y[rm], depth + 1),
        }

    def _best_split(self, X, y):
        best_gain      = float('-inf')
        best_feature   = None
        best_threshold = None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                lm = X[:, feature] <= threshold
                rm = ~lm

                if lm.sum() == 0 or rm.sum() == 0:
                    continue

                gain = self._impurity(y) \
                       - (lm.sum() / len(y)) * self._impurity(y[lm]) \
                       - (rm.sum() / len(y)) * self._impurity(y[rm])

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.array(list(Counter(y).values()))
        probs  = counts / len(y)
        if self.criterion == 'entropy':
            return -np.sum(probs * np.log2(probs + 1e-12))
        return 1.0 - np.sum(probs ** 2)   # gini

    def _traverse(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])


class AdaBoostClassifier:
    """
    AdaBoost (SAMME) — a weighted ensemble of weak learners, boosted one round at a time.

    Handles both binary and multi-class problems with the same algorithm — SAMME
    is the standard multi-class generalisation of AdaBoost, and it reduces to
    ordinary two-class AdaBoost automatically whenever there are only 2 classes.

    Each round trains one weak learner on re-weighted data, gives it a vote
    weight (alpha) based on its accuracy, then upweights the samples it got
    wrong so the next round focuses on them. Final prediction picks the class
    with the highest total alpha across every round that voted for it.

    Parameters
    ----------
    n_estimators       : int, default=50     — number of boosting rounds
    weak_learner        : str, default='stump' — 'stump' or 'tree'
    weak_learner_depth  : int, default=2      — max depth, tree learner only
    criterion            : str, default='gini' — 'gini' or 'entropy', tree learner only
    random_state         : int, default=None  — seed for reproducibility

    Attributes
    ----------
    stumps_  : list         — fitted weak learners, each carrying its own alpha
    classes_ : ndarray (n_classes,) — original class labels seen during fit
    errors_  : list          — weighted error per boosting round
    alphas_  : list          — voting weight per boosting round
    """

    def __init__(self, n_estimators=50, weak_learner='stump',
                 weak_learner_depth=2, criterion='gini', random_state=None):
        self.n_estimators       = n_estimators
        self.weak_learner       = weak_learner
        self.weak_learner_depth = weak_learner_depth
        self.criterion          = criterion
        self.random_state       = random_state

        self.stumps_  = []
        self.classes_ = None
        self.errors_  = []
        self.alphas_  = []

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,) — 2 or more classes
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

        self.classes_ = np.unique(y)
        n_classes     = len(self.classes_)

        if n_classes < 2:
            raise ValueError("AdaBoost needs at least 2 classes.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples    = X.shape[0]
        weights      = np.full(n_samples, 1 / n_samples)   # every sample starts equally important
        self.stumps_ = []
        self.errors_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            learner = self._make_learner()
            learner.fit(X, y, weights)

            predictions = learner.predict(X)
            wrong       = predictions != y
            error       = np.sum(weights[wrong])
            error       = np.clip(error, 1e-10, 1 - 1e-10)   # keep log() finite

            # SAMME alpha — reduces to ordinary AdaBoost's alpha when n_classes == 2
            alpha         = np.log((1 - error) / error) + np.log(n_classes - 1)
            learner.alpha = alpha

            weights *= np.exp(alpha * wrong)   # upweight misclassified samples
            weights /= weights.sum()

            self.stumps_.append(learner)
            self.errors_.append(float(error))
            self.alphas_.append(float(alpha))

        return self

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

        # each round adds its alpha to whichever class it voted for
        class_votes = np.zeros((X.shape[0], len(self.classes_)))
        for learner in self.stumps_:
            preds = learner.predict(X)
            for i, cls in enumerate(self.classes_):
                class_votes[:, i] += learner.alpha * (preds == cls)

        return self.classes_[np.argmax(class_votes, axis=1)]

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        return np.mean(self.predict(X_test) == np.asarray(y_test))

    def __repr__(self):
        if not self.stumps_:
            return (f"AdaBoostClassifier(n_estimators={self.n_estimators}, "
                    f"weak_learner={self.weak_learner!r})")
        return (f"AdaBoostClassifier(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  weak_learner={self.weak_learner!r},\n"
                f"  criterion={self.criterion!r},\n"
                f"  classes_={self.classes_},\n"
                f"  n_fitted={len(self.stumps_)}\n"
                f")")

    def _make_learner(self):
        if self.weak_learner == 'tree':
            return WeakDecisionTree(
                max_depth = self.weak_learner_depth,
                criterion = self.criterion
            )
        return DecisionStump()