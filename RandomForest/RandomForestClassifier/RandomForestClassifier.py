import numpy as np
from collections import Counter


class CreateNode:
    """Single node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # leaf class label

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    Parameters
    ----------
    max_depth         : int   — maximum tree depth
    min_samples_split : int   — minimum samples to split
    n_features        : int   — number of features to consider at each split
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.root              = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        majority = Counter(y).most_common(1)[0][0]

        if len(set(y)) == 1:
            return CreateNode(value=y[0])
        if len(y) < self.min_samples_split:
            return CreateNode(value=majority)
        if self.max_depth is not None and depth >= self.max_depth:
            return CreateNode(value=majority)

        feature, threshold, gain = self._best_split(X, y)

        if gain <= 0:
            return CreateNode(value=majority)

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

                gain = self._information_gain(y, y[lm], y[rm])

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.array(list(Counter(y).values()))
        probs  = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _information_gain(self, y_parent, y_left, y_right):
        n  = len(y_parent)
        nl = len(y_left)
        nr = len(y_right)
        return (self._gini(y_parent)
                - (nl / n) * self._gini(y_left)
                - (nr / n) * self._gini(y_right))

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class RandomForestClassifier:
    """
    Parameters
    ----------
    n_estimators      : int,   default=100  — number of trees
    max_depth         : int,   default=None — max depth per tree
    min_samples_split : int,   default=2    — min samples to split a node
    max_features      : str or int, default='sqrt' — features per split
                        'sqrt' = sqrt(n_features), 'log2', or int
    random_state      : int,   default=None — seed for reproducibility

    Attributes
    ----------
    trees_     : list of DecisionTree — fitted trees
    classes_   : ndarray              — unique class labels
    n_classes_ : int                  — number of classes
    n_features_: int                  — number of input features
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features='sqrt',
                 random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state

        self.trees_      = []
        self.classes_    = None
        self.n_classes_  = None
        self.n_features_ = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train).ravel()

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_    = np.unique(y)
        self.n_classes_  = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.trees_      = []

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

        return self`

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — majority vote
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)

        # collect predictions from all trees — shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        # majority vote per sample
        y_pred = []
        for sample_preds in all_preds.T:
            vote = Counter(sample_preds).most_common(1)[0][0]
            y_pred.append(vote)

        return np.array(y_pred)

    def predict_proba(self, X_test):
        """
        Output : (n_samples, n_classes) — vote fraction per class
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict_proba().")

        X        = np.asarray(X_test, dtype=np.float64)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        n_samples = X.shape[0]
        proba     = np.zeros((n_samples, self.n_classes_))

        for i, sample_preds in enumerate(all_preds.T):
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(sample_preds == cls) / self.n_estimators

        return proba

    def score(self, X_test, y_test):
        """Accuracy — fraction of correct predictions."""
        return np.mean(self.predict(X_test) == np.asarray(y_test).ravel())

    def __repr__(self):
        if not self.trees_:
            return (f"RandomForestClassifier(n_estimators={self.n_estimators}, "
                    f"max_depth={self.max_depth}, max_features={self.max_features!r})")
        return (f"RandomForestClassifier(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  max_depth={self.max_depth},\n"
                f"  max_features={self.max_features!r},\n"
                f"  n_features_={self.n_features_},\n"
                f"  n_classes_={self.n_classes_},\n"
                f"  classes_={self.classes_}\n"
                f")")

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return self.max_features
        return n_features   # None or unknown — use all features