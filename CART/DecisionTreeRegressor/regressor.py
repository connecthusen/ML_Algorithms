import numpy as np


class CreateNode:
    """Single node in the regression tree."""

    def __init__(self, feature=None, threshold=None, feature_type=None,
                 left=None, right=None, value=None):
        self.feature      = feature       # feature index to split on
        self.threshold    = threshold     # split value
        self.feature_type = feature_type  # 'numeric' or 'categorical'
        self.left         = left          # left child node
        self.right        = right         # right child node
        self.value        = value         # leaf prediction (mean of samples)

    def is_leaf(self):
        return self.value is not None

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(value={self.value:.4f})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"


class DecisionTreeRegressor:
    """
    Decision Tree Regressor supporting numeric and categorical features.
    Splits using Variance Reduction (MSE-based information gain).

    Parameters
    ----------
    max_depth         : int,   default=None — maximum tree depth (None = grow fully)
    min_samples_split : int,   default=2    — minimum samples required to split a node
    min_impurity_decrease : float, default=0.0 — minimum variance reduction to split

    Attributes
    ----------
    root      : CreateNode — root node of the fitted tree
    n_features_: int       — number of features seen during fit
    """

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0):
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root                  = None
        self.n_features_           = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_train and y_train sample count mismatch.")

        self.n_features_ = X.shape[1]
        self.root        = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,)
        """
        if self.root is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test)
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X_test, y_test):
        """R² score — how well the model explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)   # R² = 1 - SS_res/SS_tot

    def __repr__(self):
        if self.root is None:
            return (f"DecisionTreeRegressor("
                    f"max_depth={self.max_depth}, "
                    f"min_samples_split={self.min_samples_split})")
        return (f"DecisionTreeRegressor(\n"
                f"  max_depth={self.max_depth},\n"
                f"  min_samples_split={self.min_samples_split},\n"
                f"  n_features_={self.n_features_}\n"
                f")")

    # ── tree building ─────────────────────────────────────────────────────────

    def _build_tree(self, X, y, depth):
        # leaf: too few samples
        if len(y) < self.min_samples_split:
            return CreateNode(value=float(np.mean(y)))

        # leaf: max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return CreateNode(value=float(np.mean(y)))

        best_feature, best_threshold, best_feature_type, best_gain = \
            self._find_best_split(X, y)

        # leaf: no beneficial split found
        if best_gain <= self.min_impurity_decrease:
            return CreateNode(value=float(np.mean(y)))

        # split data
        if best_feature_type == 'numeric':
            left_mask  = X[:, best_feature] <= best_threshold
            right_mask = X[:, best_feature] >  best_threshold
        else:
            left_mask  = X[:, best_feature] == best_threshold
            right_mask = X[:, best_feature] != best_threshold

        left_child  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return CreateNode(
            feature      = best_feature,
            threshold    = best_threshold,
            feature_type = best_feature_type,
            left         = left_child,
            right        = right_child
        )

    def _find_best_split(self, X, y):
        best_gain         = float('-inf')
        best_feature      = None
        best_threshold    = None
        best_feature_type = None

        for feature in range(X.shape[1]):
            column = X[:, feature]

            if np.issubdtype(column.dtype, np.number):
                # numeric — try every unique value as threshold
                for threshold in np.unique(column):
                    left_mask  = column <= threshold
                    right_mask = column >  threshold

                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue

                    gain = self._variance_reduction(y, y[left_mask], y[right_mask])

                    if gain > best_gain:
                        best_gain         = gain
                        best_feature      = feature
                        best_threshold    = threshold
                        best_feature_type = 'numeric'
            else:
                # categorical — try each category as a binary split
                for category in np.unique(column):
                    left_mask  = column == category
                    right_mask = column != category

                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue

                    gain = self._variance_reduction(y, y[left_mask], y[right_mask])

                    if gain > best_gain:
                        best_gain         = gain
                        best_feature      = feature
                        best_threshold    = category
                        best_feature_type = 'categorical'

        return best_feature, best_threshold, best_feature_type, best_gain

    # ── impurity measure ──────────────────────────────────────────────────────

    def _mse(self, y):
        # MSE around the mean — used as node impurity
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - np.mean(y)) ** 2))

    def _variance_reduction(self, y_parent, y_left, y_right):
        # VR = MSE(parent) - weighted avg MSE(children)
        n  = len(y_parent)
        nl = len(y_left)
        nr = len(y_right)
        return (self._mse(y_parent)
                - (nl / n) * self._mse(y_left)
                - (nr / n) * self._mse(y_right))

    # ── prediction ────────────────────────────────────────────────────────────

    def _traverse(self, x, node):
        # walk tree until leaf — return leaf mean
        if node.is_leaf():
            return node.value

        if node.feature_type == 'numeric':
            go_left = x[node.feature] <= node.threshold
        else:
            go_left = x[node.feature] == node.threshold

        return self._traverse(x, node.left if go_left else node.right)

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self.root is None:
            raise RuntimeError("Call fit() before using predict() or score().")