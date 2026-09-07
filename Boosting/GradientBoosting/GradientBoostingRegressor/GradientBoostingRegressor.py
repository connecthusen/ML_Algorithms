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
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.root               = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

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

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                lm = X[:, feature] <= threshold
                rm = X[:, feature] >  threshold

                if lm.sum() == 0 or rm.sum() == 0:
                    continue

                gain = self._mse(y) - (lm.sum()/len(y))*self._mse(y[lm]) - (rm.sum()/len(y))*self._mse(y[rm])

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
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor — fits trees to the residual errors of squared loss.s
    Parameters
    ----------
    n_estimators      : int,   default=100 — number of boosting rounds
    learning_rate     : float, default=0.1 — shrinks each tree's contribution
    max_depth         : int,   default=3   — max depth per tree
    min_samples_split : int,   default=2   — min samples to split a node
    random_state      : int,   default=None — seed for reproducibility

    Attributes
    ----------
    trees_            : list of DecisionTree — one tree per boosting round
    init_prediction_  : float — starting prediction, the training target mean
    train_loss_       : list  — training MSE recorded after every round
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, random_state=None):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state

        self.trees_           = []
        self.init_prediction_ = None
        self.train_loss_      = []

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

        self.init_prediction_ = np.mean(y)
        current_pred           = np.full(y.shape, self.init_prediction_)
        self.trees_            = []
        self.train_loss_       = []

        for _ in range(self.n_estimators):
            residuals = y - current_pred   # negative gradient of squared loss

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)

            current_pred += self.learning_rate * tree.predict(X)

            self.trees_.append(tree)
            self.train_loss_.append(np.mean((y - current_pred) ** 2))

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — init prediction plus every tree's contribution
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        y_pred = np.full(X.shape[0], self.init_prediction_)
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def score(self, X_test, y_test):
        """R² score — how well the ensemble explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def __repr__(self):
        if not self.trees_:
            return (f"GradientBoostingRegressor(n_estimators={self.n_estimators}, "
                    f"learning_rate={self.learning_rate}, max_depth={self.max_depth})")
        return (f"GradientBoostingRegressor(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  learning_rate={self.learning_rate},\n"
                f"  max_depth={self.max_depth},\n"
                f"  init_prediction_={self.init_prediction_:.4f},\n"
                f"  n_fitted={len(self.trees_)}\n"
                f")")
