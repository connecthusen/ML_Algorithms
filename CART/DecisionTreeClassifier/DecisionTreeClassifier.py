import numpy as np
from collections import Counter


class CreateNode:
    """Single node in the decision tree."""

    def __init__(self, feature=None, threshold=None, feature_type=None,
                 left=None, right=None, value=None):
        self.feature      = feature       # feature index to split on
        self.threshold    = threshold     # split value
        self.feature_type = feature_type  # 'numeric' or 'categorical'
        self.left         = left          # left child node
        self.right        = right         # right child node
        self.value        = value         # leaf class label (set if leaf)

    def is_leaf(self):
        return self.value is not None

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(value={self.value})"
        return f"Node(feature={self.feature}, threshold={self.threshold})"


class DecisionTreeClassifier:
    """
    Decision Tree Classifier supporting numeric and categorical features.
    Splits using Information Gain with Gini or Entropy impurity.

    Parameters
    ----------
    max_depth         : int,   default=None  — maximum tree depth (None = grow fully)
    criterion         : str,   default='gini' — 'gini' or 'entropy'
    min_samples_split : int,   default=2     — minimum samples required to split a node

    Attributes
    ----------
    root      : CreateNode — root node of the fitted tree
    n_classes_: int        — number of unique classes
    classes_  : ndarray    — unique class labels
    """

    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2):
        self.max_depth         = max_depth
        self.impurity          = criterion
        self.min_samples_split = min_samples_split
        self.root              = None
        self.classes_          = None
        self.n_classes_        = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train)
        y = np.asarray(y_train).ravel()

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.root       = self._build_tree(X, y, depth=0)

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
        """Accuracy — fraction of correct predictions."""
        return np.mean(self.predict(X_test) == np.asarray(y_test).ravel())

    def __repr__(self):
        if self.root is None:
            return (f"DecisionTreeClassifier(criterion={self.impurity!r}, "
                    f"max_depth={self.max_depth})")
        return (f"DecisionTreeClassifier(\n"
                f"  criterion={self.impurity!r},\n"
                f"  max_depth={self.max_depth},\n"
                f"  n_classes_={self.n_classes_},\n"
                f"  classes_={self.classes_}\n"
                f")")

    # tree building 

    def _build_tree(self, X, y, depth):
        majority_class = Counter(y).most_common(1)[0][0]

        # stopping conditions → return leaf
        if len(set(y)) == 1:
            return CreateNode(value=y[0])
        if len(y) < self.min_samples_split:
            return CreateNode(value=majority_class)
        if self.max_depth is not None and depth >= self.max_depth:
            return CreateNode(value=majority_class)

        best_feature, best_threshold, best_feature_type, best_gain = \
            self._find_best_split(X, y)

        if best_gain <= 0:
            return CreateNode(value=majority_class)

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

                    gain = self._information_gain(y, y[left_mask], y[right_mask])

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

                    gain = self._information_gain(y, y[left_mask], y[right_mask])

                    if gain > best_gain:
                        best_gain         = gain
                        best_feature      = feature
                        best_threshold    = category
                        best_feature_type = 'categorical'

        return best_feature, best_threshold, best_feature_type, best_gain

    #  impurity measures 

    def _gini(self, y):
        # Gini = 1 - sum(p_k^2)
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.array(list(Counter(y).values()))
        probs  = counts / m
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        # Entropy = -sum(p_k * log2(p_k))
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.array(list(Counter(y).values()))
        probs  = counts / m
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _impurity(self, y):
        if self.impurity == 'entropy':
            return self._entropy(y)
        return self._gini(y)

    def _information_gain(self, y_parent, y_left, y_right):
        # IG = impurity(parent) - weighted avg impurity(children)
        n  = len(y_parent)
        nl = len(y_left)
        nr = len(y_right)
        return (self._impurity(y_parent)
                - (nl / n) * self._impurity(y_left)
                - (nr / n) * self._impurity(y_right))

    #  prediction

    def _traverse(self, x, node):
        # recursively walk the tree until a leaf
        if node.is_leaf():
            return node.value

        if node.feature_type == 'numeric':
            go_left = x[node.feature] <= node.threshold
        else:
            go_left = x[node.feature] == node.threshold

        return self._traverse(x, node.left if go_left else node.right)
