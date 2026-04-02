import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.5, n_iterations=2000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        # Insert bias column (column of 1s)
        X_train = np.insert(X_train, 0, 1, axis=1)
        weights = np.zeros(X_train.shape[1])

        for i in range(self.n_iterations):
            y_hat = self.sigmoid(np.dot(X_train, weights))
            weights += self.lr * (np.dot((y_train - y_hat), X_train) / X_train.shape[0])

        self.weights = weights[1:]  # Feature weights
        self.bias = weights[0]      # Bias term

    def predict_proba(self, X_test):
        X_test = np.asarray(X_test)
        return self.sigmoid(np.dot(X_test, self.weights) + self.bias)

    def predict(self, X_test, threshold=0.5):
        return (self.predict_proba(X_test) >= threshold).astype(int)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.mean(predictions == np.asarray(y_test))