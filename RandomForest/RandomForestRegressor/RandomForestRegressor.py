import numpy as np


class CreateNode:
    """Single node in a regression tree."""

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # leaf mean prediction

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Parameters
    ----------
    max_depth         : int — maximum tree depth
    min_samples_split : int — minimum samples to split
    n_features        : int — number of features to consider at each split
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth            = max_depth
        self.min_samples_split    = min_samples_split
        self.n_features           = n_features
        self.root                 = None
        self.feature_importances_ = None   # (n_features_total,) — variance reduction per feature

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build_tree(X, y, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        mean_value = np.mean(y)

        if np.var(y) == 0:
            return CreateNode(value=mean_value)
        if len(y) < self.min_samples_split:
            return CreateNode(value=mean_value)
        if self.max_depth is not None and depth >= self.max_depth:
            return CreateNode(value=mean_value)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            return CreateNode(value=mean_value)

        self.feature_importances_[feature] += gain * len(y)   # weight by node size

        left_mask  = X[:, feature] <= threshold
        right_mask = X[:, feature] >  threshold

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return CreateNode(feature=feature, threshold=threshold,
                          left=left, right=right)

    def _best_split(self, X, y):
        best_gain      = float('-inf')
        best_feature   = None
        best_threshold = None

        # random feature subset — the key to Random Forest diversity
        n_features_total = X.shape[1]
        n_features       = self.n_features or n_features_total
        feature_indices  = np.random.choice(n_features_total, n_features, replace=False)

        for feature in feature_indices:
            for threshold in np.unique(X[:, feature]):
                lm = X[:, feature] <= threshold
                rm = X[:, feature] >  threshold

                if lm.sum() == 0 or rm.sum() == 0:
                    continue

                gain = self._variance_reduction(y, y[lm], y[rm])

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return np.var(y)   # spread around the mean

    def _variance_reduction(self, y_parent, y_left, y_right):
        n  = len(y_parent)
        nl = len(y_left)
        nr = len(y_right)
        return (self._mse(y_parent)
                - (nl / n) * self._mse(y_left)
                - (nr / n) * self._mse(y_right))

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class RandomForestRegressor:
    """
    Parameters
    ----------
    n_estimators      : int,   default=100  — number of trees
    max_depth         : int,   default=None — max depth per tree
    min_samples_split : int,   default=2    — min samples to split a node
    max_features      : str, int or float, default=1.0 — features per split
                        'sqrt' = sqrt(n_features), 'log2', int, or fraction
    random_state      : int,   default=None — seed for reproducibility

    Attributes
    ----------
    trees_               : list of DecisionTree — fitted trees
    n_features_          : int                  — number of input features
    feature_importances_ : ndarray (n_features_,) — variance reduction per feature, averaged across trees
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features=1.0,
                 random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state

        self.trees_               = []
        self.n_features_          = None
        self.feature_importances_ = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_ = X.shape[1]
        self.trees_       = []

        n_features_split = self._resolve_max_features(self.n_features_)

        for _ in range(self.n_estimators):
            # bootstrap sample
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_boot  = X[indices]
            y_boot  = y[indices]

            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features        = n_features_split
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        # average each tree's normalised importances across the forest
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — mean of all tree predictions
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)

        # collect predictions from all trees — shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        return np.mean(all_preds, axis=0)

    def score(self, X_test, y_test):
        """R² score — how well the forest explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def __repr__(self):
        if not self.trees_:
            return (f"RandomForestRegressor(n_estimators={self.n_estimators}, "
                    f"max_depth={self.max_depth}, max_features={self.max_features!r})")
        return (f"RandomForestRegressor(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  max_depth={self.max_depth},\n"
                f"  max_features={self.max_features!r},\n"
                f"  n_features_={self.n_features_}\n"
                f")")

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return self.max_features
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return n_features   # None or unknown — use all features
