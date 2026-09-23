import numpy as np


class CreateNode:
    """Single node in a classification tree."""

    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None, gain=0.0):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value   # leaf output score
        self.gain      = gain

    def is_leaf(self):
        return self.value is not None


class XGBTree:
    """
    Single regression tree used inside XGBoostClassifier.
    Outputs raw scores (logits) — not probabilities.

    Parameters
    ----------
    max_depth         : int   — maximum tree depth
    min_samples_split : int   — minimum samples to split
    reg_lambda        : float — L2 regularisation on leaf weights
    gamma             : float — minimum gain to allow a split
    """

    def __init__(self, max_depth=3, min_samples_split=2,
                 reg_lambda=1.0, gamma=0.0):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda        = reg_lambda
        self.gamma             = gamma
        self.root              = None

    def fit(self, X, gradients, hessians):
        self.root = self._build(X, gradients, hessians, depth=0)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _build(self, X, g, h, depth):
        # optimal leaf weight: -G / (H + lambda)
        leaf_value = -np.sum(g) / (np.sum(h) + self.reg_lambda)

        if len(g) < self.min_samples_split:
            return CreateNode(value=leaf_value)
        if depth >= self.max_depth:
            return CreateNode(value=leaf_value)

        feature, threshold, gain = self._best_split(X, g, h)

        if feature is None or gain < self.gamma:
            return CreateNode(value=leaf_value)

        lm = X[:, feature] <= threshold
        rm = ~lm

        left  = self._build(X[lm],  g[lm],  h[lm],  depth + 1)
        right = self._build(X[rm], g[rm], h[rm], depth + 1)

        return CreateNode(feature=feature, threshold=threshold,
                          left=left, right=right, gain=gain)

    def _best_split(self, X, g, h):
        best_gain      = float('-inf')
        best_feature   = None
        best_threshold = None

        G = np.sum(g)
        H = np.sum(h)

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                lm = X[:, feature] <= threshold
                rm = ~lm

                if lm.sum() == 0 or rm.sum() == 0:
                    continue

                G_l, H_l = np.sum(g[lm]), np.sum(h[lm])
                G_r, H_r = G - G_l, H - H_l

                gain = 0.5 * (
                    G_l**2 / (H_l + self.reg_lambda) +
                    G_r**2 / (H_r + self.reg_lambda) -
                    G**2  / (H  + self.reg_lambda)
                ) - self.gamma

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class XGBoostClassifier:
    """
    XGBoost Classifier — second-order gradient boosting for classification.

    Supports binary and multiclass (One-vs-Rest) classification.
    Uses log-loss for binary: g = p - y, h = p(1-p).
    Uses softmax cross-entropy for multiclass: OvR with one tree per class.

    Parameters
    ----------
    n_estimators      : int,   default=100  — number of boosting rounds
    learning_rate     : float, default=0.1  — shrinks each tree's contribution
    max_depth         : int,   default=3    — max depth per tree
    min_samples_split : int,   default=2    — min samples to split a node
    reg_lambda        : float, default=1.0  — L2 regularisation on leaf weights
    gamma             : float, default=0.0  — min gain to allow a split
    random_state      : int,   default=None — seed for reproducibility

    Attributes
    ----------
    trees_            : list — fitted trees per round (list of lists for multiclass)
    classes_          : ndarray — unique class labels
    n_classes_        : int    — number of classes
    init_score_       : float or ndarray — initial log-odds / class scores
    train_loss_       : list   — training log-loss after every round
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0, gamma=0.0,
                 random_state=None):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda        = reg_lambda
        self.gamma             = gamma
        self.random_state      = random_state

        self.trees_      = []
        self.classes_    = None
        self.n_classes_  = None
        self.init_score_ = None
        self.train_loss_ = []

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,) — any binary or multiclass labels
        """
        X     = np.asarray(X_train, dtype=np.float64)
        y_raw = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y_raw.shape[0]:
            raise ValueError(f"X and y sample count mismatch.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_   = np.unique(y_raw)
        self.n_classes_ = len(self.classes_)
        self.trees_     = []
        self.train_loss_ = []

        if self.n_classes_ == 2:
            self._fit_binary(X, y_raw)
        else:
            self._fit_multiclass(X, y_raw)

        return self

    def predict_proba(self, X_test):
        """
        Output : (n_samples, n_classes) — class probabilities
        """
        if not self.trees_:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X_test, dtype=np.float64)

        if self.n_classes_ == 2:
            p = self._sigmoid(self._raw_scores_binary(X))
            return np.column_stack([1 - p, p])
        else:
            return self._softmax(self._raw_scores_multiclass(X))

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — predicted class labels
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X_test, y_test):
        """Accuracy — fraction of correctly classified samples."""
        return np.mean(self.predict(X_test) == np.asarray(y_test))

    def __repr__(self):
        if not self.trees_:
            return (f"XGBoostClassifier(n_estimators={self.n_estimators}, "
                    f"learning_rate={self.learning_rate}, max_depth={self.max_depth}, "
                    f"reg_lambda={self.reg_lambda}, gamma={self.gamma})")
        return (f"XGBoostClassifier(\n"
                f"  n_estimators={self.n_estimators},\n"
                f"  learning_rate={self.learning_rate},\n"
                f"  max_depth={self.max_depth},\n"
                f"  reg_lambda={self.reg_lambda},\n"
                f"  gamma={self.gamma},\n"
                f"  classes_={self.classes_},\n"
                f"  n_classes_={self.n_classes_},\n"
                f"  n_fitted={len(self.trees_)}\n"
                f")")

    #  binary

    def _fit_binary(self, X, y_raw):
        # map labels to 0/1
        y = (y_raw == self.classes_[1]).astype(np.float64)

        # init score: log-odds of positive class
        p0 = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        self.init_score_ = np.log(p0 / (1 - p0))
        raw_scores       = np.full(len(y), self.init_score_)

        for _ in range(self.n_estimators):
            p = self._sigmoid(raw_scores)

            # log-loss gradients and hessians
            g = p - y
            h = p * (1 - p)

            tree = XGBTree(max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, g, h)

            raw_scores += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

            # log-loss for monitoring
            p_clip = np.clip(self._sigmoid(raw_scores), 1e-7, 1 - 1e-7)
            loss   = -np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
            self.train_loss_.append(float(loss))

    # ── multiclass ────────────────────────────────────────────────────────────

    def _fit_multiclass(self, X, y_raw):
        K = self.n_classes_
        n = len(y_raw)

        # one-hot encode
        Y = np.zeros((n, K))
        for k, cls in enumerate(self.classes_):
            Y[:, k] = (y_raw == cls).astype(np.float64)

        # init scores: log of class frequencies
        class_counts     = Y.sum(axis=0)
        self.init_score_ = np.log(class_counts / n + 1e-7)
        raw_scores       = np.tile(self.init_score_, (n, 1))   # (n, K)

        for _ in range(self.n_estimators):
            probs = self._softmax(raw_scores)   # (n, K)

            round_trees = []
            for k in range(K):
                g = probs[:, k] - Y[:, k]       # softmax gradient
                h = probs[:, k] * (1 - probs[:, k])   # softmax hessian

                tree = XGBTree(max_depth=self.max_depth,
                               min_samples_split=self.min_samples_split,
                               reg_lambda=self.reg_lambda, gamma=self.gamma)
                tree.fit(X, g, h)
                raw_scores[:, k] += self.learning_rate * tree.predict(X)
                round_trees.append(tree)

            self.trees_.append(round_trees)

            # cross-entropy loss for monitoring
            p_clip = np.clip(self._softmax(raw_scores), 1e-7, 1 - 1e-7)
            loss   = -np.mean(np.sum(Y * np.log(p_clip), axis=1))
            self.train_loss_.append(float(loss))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _raw_scores_binary(self, X):
        scores = np.full(X.shape[0], self.init_score_)
        for tree in self.trees_:
            scores += self.learning_rate * tree.predict(X)
        return scores

    def _raw_scores_multiclass(self, X):
        n = X.shape[0]
        scores = np.tile(self.init_score_, (n, 1))
        for round_trees in self.trees_:
            for k, tree in enumerate(round_trees):
                scores[:, k] += self.learning_rate * tree.predict(X)
        return scores

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)   # numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _check_is_fitted(self):
        if not self.trees_:
            raise RuntimeError("Call fit() before using predict() or score().")
