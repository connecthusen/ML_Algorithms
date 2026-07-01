import numpy as np


class LogisticRegression:
    """
    Logistic Regression — Binary (sigmoid/OvR) and Multinomial (softmax).
    Supports L1, L2, ElasticNet, and no regularization.

    Parameters
    ----------
    lr            : float, default=0.5    — learning rate
    n_iterations  : int,   default=2000  — gradient descent steps
    multi_class   : str,   default='ovr' — 'ovr' (sigmoid) | 'multinomial' (softmax)
    penalty       : str,   default='l2'  — 'none' | 'l2' | 'l1' | 'elasticnet'
    C             : float, default=1.0   — inverse regularization strength (larger = weaker)
    l1_ratio      : float, default=0.5   — ElasticNet mix; 0=pure L2, 1=pure L1

    Attributes
    ----------
    weights  : ndarray (n_features,) or (n_features, n_classes) — learned weights
    bias     : float or ndarray (n_classes,)                    — learned bias
    classes_ : ndarray — unique class labels from y_train
    """

    def __init__(self, lr=0.5, n_iterations=2000, multi_class="ovr",
                 penalty="l2", C=1.0, l1_ratio=0.5):
        self.lr           = lr
        self.n_iterations = n_iterations
        self.multi_class  = multi_class
        self.penalty      = penalty
        self.C            = C          # lambda_eff = 1 / (C * n_samples)
        self.l1_ratio     = l1_ratio
        self.weights      = None
        self.bias         = None
        self.classes_     = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z     = z - np.max(z, axis=1, keepdims=True)   # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        n_classes = len(self.classes_)
        one_hot   = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            one_hot[i, np.where(self.classes_ == label)[0][0]] = 1
        return one_hot

    def _penalty_gradient(self, w, n_samples):
        # gradient of regularization term w.r.t. weights — bias excluded
        lam = 1.0 / (self.C * n_samples)

        if self.penalty == "l2":
            return lam * w

        if self.penalty == "l1":
            return lam * np.sign(w)

        if self.penalty == "elasticnet":
            return lam * (self.l1_ratio * np.sign(w) + (1 - self.l1_ratio) * w)

        return 0.0   # penalty == 'none'

    def _fit_binary(self, X_train, y_train):
        n_samples, n_features = X_train.shape

        # prepend bias column so bias is solved alongside weights
        X_aug   = np.insert(X_train, 0, 1, axis=1)   # (n, 1+p)
        weights = np.zeros(X_aug.shape[1])            # [bias, w1, ..., wp]

        for _ in range(self.n_iterations):
            y_hat = self.sigmoid(X_aug @ weights)
            grad  = X_aug.T @ (y_train - y_hat) / n_samples   # data gradient

            # penalty applies only to weights (index 1:), not bias (index 0)
            pen_grad     = np.zeros_like(weights)
            pen_grad[1:] = self._penalty_gradient(weights[1:], n_samples)

            weights += self.lr * (grad - pen_grad)

        self.bias    = weights[0]
        self.weights = weights[1:]

    def _fit_multinomial(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = len(self.classes_)

        Y_one_hot    = self._one_hot_encode(y_train)          # (n, K)
        self.weights = np.zeros((n_features, n_classes))      # (p, K)
        self.bias    = np.zeros(n_classes)                    # (K,)

        for _ in range(self.n_iterations):
            logits = X_train @ self.weights + self.bias       # (n, K)
            y_hat  = self.softmax(logits)                     # (n, K)
            error  = y_hat - Y_one_hot                        # (n, K)

            w_grad = X_train.T @ error / n_samples            # (p, K)
            b_grad = np.mean(error, axis=0)                   # (K,)

            pen_grad = self._penalty_gradient(self.weights, n_samples)

            self.weights -= self.lr * (w_grad + pen_grad)
            self.bias    -= self.lr * b_grad

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X_train       = np.asarray(X_train)
        y_train       = np.asarray(y_train)
        self.classes_ = np.unique(y_train)

        if self.multi_class == "multinomial":
            self._fit_multinomial(X_train, y_train)
        else:
            self._fit_binary(X_train, y_train)

        return self

    def predict_proba(self, X_test):
        """
        Output : (n_samples,) for binary | (n_samples, n_classes) for multinomial
        """
        X_test = np.asarray(X_test)
        if self.multi_class == "multinomial":
            return self.softmax(X_test @ self.weights + self.bias)
        return self.sigmoid(X_test @ self.weights + self.bias)

    def predict(self, X_test, threshold=0.5):
        """
        Output : predicted class labels (n_samples,)
        """
        X_test = np.asarray(X_test)
        if self.multi_class == "multinomial":
            return self.classes_[np.argmax(self.predict_proba(X_test), axis=1)]
        return (self.predict_proba(X_test) >= threshold).astype(int)

    def score(self, X_test, y_test):
        """Accuracy — fraction of correct predictions."""
        return np.mean(self.predict(X_test) == np.asarray(y_test))