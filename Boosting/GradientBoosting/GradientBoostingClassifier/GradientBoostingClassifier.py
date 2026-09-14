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


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier — fits trees to the gradient of log loss.

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
    classes_          : ndarray (2,) — original class labels seen during fit
    init_prediction_  : float — starting raw score, the log-odds of the class balance
    train_loss_       : list  — training log loss recorded after every round
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, random_state=None):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state

        self.trees_           = []
        self.classes_         = None
        self.init_prediction_ = None
        self.train_loss_      = []

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

        self.classes_ = np.unique(y_raw)
        if len(self.classes_) != 2:
            raise ValueError("GradientBoostingClassifier only supports binary classification.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        y = np.where(y_raw == self.classes_[0], 0, 1).astype(np.float64)   # map to {0, 1}

        p = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        self.init_prediction_ = np.log(p / (1 - p))   # starting log-odds
        current_raw            = np.full(y.shape, self.init_prediction_)
        self.trees_            = []
        self.train_loss_       = []

        for _ in range(self.n_estimators):
            proba     = self._sigmoid(current_raw)
            residuals = y - proba   # negative gradient of log loss

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)

            current_raw += self.learning_rate * tree.predict(X)

            self.trees_.append(tree)
            self.train_loss_.append(self._log_loss(y, self._sigmoid(current_raw)))

        return self

    def predict_proba(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : (n_samples, 2) — probability of each class, columns match classes_
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        raw = np.full(X.shape[0], self.init_prediction_)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(X)

        p1 = self._sigmoid(raw)
        return np.column_stack([1 - p1, p1])

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — original class labels
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        y_pred = self.predict(X_test)
        return np.mean(y_pred == np.asarray(y_test))

    def __repr__(self):
        if not self.trees_:
            return (f"GradientBoostingClassifier(n_estimators={self.n_estimators}, "
                    f"learning_rate={self.learning_rate}, max_depth={self.max_depth})")
        return (f"GradientBoostingClassifier(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  learning_rate={self.learning_rate},\n"
                f"  max_depth={self.max_depth},\n"
                f"  classes_={self.classes_},\n"
                f"  init_prediction_={self.init_prediction_:.4f},\n"
                f"  n_fitted={len(self.trees_)}\n"
                f")")

    def _sigmoid(self, raw):
        return 1 / (1 + np.exp(-raw))

    def _log_loss(self, y, proba):
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))
