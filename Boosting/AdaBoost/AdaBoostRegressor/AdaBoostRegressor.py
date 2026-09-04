import numpy as np


class DecisionStumpRegressor:
    """Depth-1 regression tree — a simple, fast AdaBoost.R2 weak learner."""

    def __init__(self):
        self.feature      = None
        self.threshold    = None
        self.left_value   = None   # predicted value if x[feature] <= threshold
        self.right_value  = None   # predicted value if x[feature] > threshold

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float('inf')

        for feature in range(n_features):
            for threshold in np.unique(X[:, feature]):
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_value  = self._weighted_mean(y[left_mask],  weights[left_mask])
                right_value = self._weighted_mean(y[right_mask], weights[right_mask])

                pred  = np.where(left_mask, left_value, right_value)
                error = np.sum(weights * (y - pred) ** 2)

                if error < best_error:
                    best_error       = error
                    self.feature     = feature
                    self.threshold   = threshold
                    self.left_value  = left_value
                    self.right_value = right_value

    def predict(self, X):
        column = X[:, self.feature]
        return np.where(column <= self.threshold, self.left_value, self.right_value)

    def _weighted_mean(self, y_subset, w_subset):
        """The sample-weighted average target value in this subset."""
        if len(y_subset) == 0 or w_subset.sum() == 0:
            return 0.0
        return np.sum(y_subset * w_subset) / np.sum(w_subset)


class WeakDecisionTreeRegressor:
    """A shallow regression tree weak learner, for when stumps underfit the data."""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
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
        mean_value = np.mean(y)

        if np.var(y) == 0:
            return {'leaf': True, 'value': mean_value}
        if depth >= self.max_depth or len(y) < 2:
            return {'leaf': True, 'value': mean_value}

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            return {'leaf': True, 'value': mean_value}

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

                gain = self._mse(y) \
                       - (lm.sum() / len(y)) * self._mse(y[lm]) \
                       - (rm.sum() / len(y)) * self._mse(y[rm])

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return np.var(y)   # spread around the mean

    def _traverse(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])


class AdaBoostRegressor:
    """
    AdaBoost.R2 — a weighted ensemble of weak regressors, boosted one round at a time.

    Parameters
    ----------
    n_estimators       : int, default=50      — number of boosting rounds
    weak_learner        : str, default='stump'  — 'stump' or 'tree'
    weak_learner_depth  : int, default=3        — max depth, tree learner only
    loss                 : str, default='linear' — 'linear', 'square', or 'exponential'
    random_state         : int, default=None    — seed for reproducibility

    Attributes
    ----------
    estimators_ : list  — fitted weak learners
    weights_    : list  — this round's confidence, used when combining predictions
    errors_     : list  — average weighted loss per boosting round
    """

    def __init__(self, n_estimators=50, weak_learner='stump',
                 weak_learner_depth=3, loss='linear', random_state=None):
        self.n_estimators       = n_estimators
        self.weak_learner       = weak_learner
        self.weak_learner_depth = weak_learner_depth
        self.loss               = loss
        self.random_state       = random_state

        self.estimators_ = []
        self.weights_    = []
        self.errors_     = []

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples         = X.shape[0]
        weights            = np.full(n_samples, 1 / n_samples)   # every sample starts equally important
        self.estimators_  = []
        self.weights_      = []
        self.errors_       = []

        for _ in range(self.n_estimators):
            learner = self._make_learner()
            learner.fit(X, y, weights)

            predictions = learner.predict(X)
            abs_errors  = np.abs(predictions - y)
            max_error   = abs_errors.max()

            if max_error == 0:
                self.estimators_.append(learner)   # perfect fit — trust it fully, stop early
                self.weights_.append(1.0)
                self.errors_.append(0.0)
                break

            sample_loss = self._sample_loss(abs_errors, max_error)
            avg_loss    = np.sum(weights * sample_loss)

            if avg_loss >= 0.5:
                if not self.estimators_:
                    self.estimators_.append(learner)   # keep at least one round, even if weak
                    self.weights_.append(1e-6)
                    self.errors_.append(float(avg_loss))
                break   # this round is no better than chance — stop boosting

            beta         = avg_loss / (1 - avg_loss)
            model_weight = np.log(1 / beta)

            weights *= beta ** (1 - sample_loss)   # shrink weight for well-predicted samples
            weights /= weights.sum()

            self.estimators_.append(learner)
            self.weights_.append(model_weight)
            self.errors_.append(float(avg_loss))

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — weighted median across all rounds
        """
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        all_preds     = np.array([est.predict(X) for est in self.estimators_])   # (T, n_samples)
        model_weights = np.array(self.weights_)

        return np.array([
            self._weighted_median(all_preds[:, i], model_weights)
            for i in range(X.shape[0])
        ])

    def score(self, X_test, y_test):
        """R² score — how well the ensemble explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def __repr__(self):
        if not self.estimators_:
            return (f"AdaBoostRegressor(n_estimators={self.n_estimators}, "
                    f"weak_learner={self.weak_learner!r})")
        return (f"AdaBoostRegressor(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  weak_learner={self.weak_learner!r},\n"
                f"  loss={self.loss!r},\n"
                f"  n_fitted={len(self.estimators_)}\n"
                f")")

    def _sample_loss(self, abs_errors, max_error):
        scaled = abs_errors / max_error
        if self.loss == 'square':
            return scaled ** 2
        if self.loss == 'exponential':
            return 1 - np.exp(-scaled)
        return scaled   # linear

    def _weighted_median(self, values, weights):
        """The smallest prediction where cumulative model weight first reaches half the total."""
        order       = np.argsort(values)
        sorted_vals = values[order]
        cum_weights = np.cumsum(weights[order])
        cutoff      = 0.5 * cum_weights[-1]
        return sorted_vals[np.searchsorted(cum_weights, cutoff)]

    def _make_learner(self):
        if self.weak_learner == 'tree':
            return WeakDecisionTreeRegressor(max_depth=self.weak_learner_depth)
        return DecisionStumpRegressor()
