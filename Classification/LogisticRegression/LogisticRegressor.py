import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.5, n_iterations=2000, multi_class='ovr'):
        self.lr = lr
        self.n_iterations = n_iterations
        self.multi_class = multi_class
        self.weights = None
        self.bias = None
        self.classes_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        # Subtract max for numerical stability
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        n_classes = len(self.classes_)
        one_hot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            one_hot[i, np.where(self.classes_ == label)[0][0]] = 1
        return one_hot

    # ── Binary fit ───────────────────────────────────────────────────────
    def _fit_binary(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        weights = np.zeros(X_train.shape[1])

        for _ in range(self.n_iterations):
            y_hat = self.sigmoid(np.dot(X_train, weights))
            weights += self.lr * (np.dot((y_train - y_hat), X_train) / X_train.shape[0])

        self.weights = weights[1:]
        self.bias = weights[0]

    # ── Multinomial fit (Softmax Regression) ─────────────────────────────
    def _fit_multinomial(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = len(self.classes_)

        Y_one_hot = self._one_hot_encode(y_train)           # (n_samples, n_classes)
        self.weights = np.zeros((n_features, n_classes))    # (n_features, n_classes)
        self.bias    = np.zeros(n_classes)                  # (n_classes,)

        for _ in range(self.n_iterations):
            logits = np.dot(X_train, self.weights) + self.bias  # (n_samples, n_classes)
            y_hat  = self.softmax(logits)                        # (n_samples, n_classes)
            error  = y_hat - Y_one_hot                          # (n_samples, n_classes)

            # Gradient descent
            self.weights -= self.lr * np.dot(X_train.T, error) / n_samples
            self.bias    -= self.lr * np.mean(error, axis=0)

    # ── Public API ───────────────────────────────────────────────────────
    def fit(self, X_train, y_train):
        X_train      = np.asarray(X_train)
        y_train      = np.asarray(y_train)
        self.classes_ = np.unique(y_train)

        if self.multi_class == 'multinomial':
            self._fit_multinomial(X_train, y_train)
        else:                                   # default: binary / OvR
            self._fit_binary(X_train, y_train)

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test)
        if self.multi_class == 'multinomial':
            logits = np.dot(X_test, self.weights) + self.bias
            return self.softmax(logits)         # (n_samples, n_classes)
        else:
            return self.sigmoid(np.dot(X_test, self.weights) + self.bias)

    def predict(self, X_test, threshold=0.5):
        X_test = np.asarray(X_test)
        if self.multi_class == 'multinomial':
            proba = self.predict_proba(X_test)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            return (self.predict_proba(X_test) >= threshold).astype(int)

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == np.asarray(y_test))