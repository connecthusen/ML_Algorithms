import numpy as np


class KNearestNeighbors:
    """
    K-Nearest Neighbors — Classification & Regression.

    Parameters
    ----------
    k             : int   – number of nearest neighbors to consider (default 5)
    task          : str   – 'classification' | 'regression'
    metric        : str   – 'euclidean' | 'manhattan' | 'minkowski' | 'cosine'
    p             : float – Minkowski order (p=1 → Manhattan, p=2 → Euclidean)
    weights       : str   – 'uniform' (equal vote) | 'distance' (inverse-distance)
    """

    def __init__(
        self,
        k=5,
        task="classification",
        metric="euclidean",
        p=2,
        weights="uniform",
    ):
        self.k       = k
        self.task    = task
        self.metric  = metric
        self.p       = p
        self.weights = weights

        self._X_train = None
        self._y_train = None

    #  Distance metrics                                                    

    def _euclidean(self, X):
        """‖xᵢ − xⱼ‖₂  =  √ Σⱼ (xᵢⱼ − xⱼⱼ)²"""
        diff = self._X_train[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n_train, n_test, p)
        return np.sqrt(np.sum(diff ** 2, axis=2))                      # (n_train, n_test)

    def _manhattan(self, X):
        """‖xᵢ − xⱼ‖₁  =  Σⱼ |xᵢⱼ − xⱼⱼ|"""
        diff = self._X_train[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sum(np.abs(diff), axis=2)

    def _minkowski(self, X):
        """‖xᵢ − xⱼ‖ₚ  =  (Σⱼ |xᵢⱼ − xⱼⱼ|ᵖ)^(1/p)"""
        diff = self._X_train[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sum(np.abs(diff) ** self.p, axis=2) ** (1.0 / self.p)

    def _cosine(self, X):
        """
        Cosine distance  =  1 − cos(θ)
        cos(θ) = (xᵢ · xⱼ) / (‖xᵢ‖ · ‖xⱼ‖)

        Clips distance to [0, 2] to handle floating-point rounding near ±1.
        """
        dot        = self._X_train @ X.T                                        # (n_train, n_test)
        norm_train = np.linalg.norm(self._X_train, axis=1, keepdims=True)       # (n_train, 1)
        norm_test  = np.linalg.norm(X, axis=1, keepdims=True).T                 # (1, n_test)
        cosine_sim = dot / (norm_train * norm_test + 1e-12)
        return np.clip(1.0 - cosine_sim, 0.0, 2.0)

    def _compute_distances(self, X):
        dispatch = {
            "euclidean" : self._euclidean,
            "manhattan" : self._manhattan,
            "minkowski" : self._minkowski,
            "cosine"    : self._cosine,
        }
        if self.metric not in dispatch:
            raise ValueError(f"Unknown metric '{self.metric}'. "
                             f"Choose from: {list(dispatch.keys())}")
        return dispatch[self.metric](X)   # (n_train, n_test)

    #  Weighted voting helpers                                             

    def _get_weights(self, distances):
        """
        Return per-neighbor weights shaped (n_test, k).

        uniform  : all weights = 1
        distance : wᵢ = 1 / dᵢ  (exact-match neighbors get weight ∞ → handled
                   by assigning 1.0 to that neighbor and 0.0 to all others)
        """
        if self.weights == "uniform":
            return np.ones_like(distances)

        # Inverse-distance weighting
        w = np.zeros_like(distances, dtype=float)
        for i, row in enumerate(distances):
            zero_mask = row == 0.0
            if zero_mask.any():
                w[i, zero_mask] = 1.0       
            else:
                w[i] = 1.0 / row            
        return w

    #  Prediction helpers                                                  

    def _predict_single_classification(self, neighbor_labels, neighbor_weights):
        """Weighted majority vote over k neighbor labels."""
        classes, inverse = np.unique(neighbor_labels, return_inverse=True)
        weighted_counts  = np.bincount(inverse, weights=neighbor_weights,
                                       minlength=len(classes))
        return classes[np.argmax(weighted_counts)]

    def _predict_single_regression(self, neighbor_values, neighbor_weights):
        """Weighted mean of k neighbor target values."""
        total_weight = np.sum(neighbor_weights)
        if total_weight == 0:
            return np.mean(neighbor_values)
        return np.dot(neighbor_weights, neighbor_values) / total_weight

    #  Public API                                                          

    def fit(self, X_train, y_train):
        """
        Store the training set — KNN is a lazy learner; no computation is
        performed at fit time beyond input validation.
        """
        self._X_train = np.asarray(X_train, dtype=float)
        self._y_train = np.asarray(y_train)
        return self

    def predict(self, X_test):
        X_test    = np.asarray(X_test, dtype=float)
        dist_mat  = self._compute_distances(X_test)   # (n_train, n_test)

        # k nearest neighbors for every test sample: indices into _X_train
        knn_idx   = np.argsort(dist_mat, axis=0)[:self.k, :].T  # (n_test, k)
        knn_dists = dist_mat[knn_idx, np.arange(X_test.shape[0])[:, None]]     # (n_test, k)
        knn_labels= self._y_train[knn_idx]                                      # (n_test, k)
        knn_w     = self._get_weights(knn_dists)                                # (n_test, k)

        if self.task == "classification":
            return np.array([
                self._predict_single_classification(knn_labels[i], knn_w[i])
                for i in range(X_test.shape[0])
            ])
        else:
            return np.array([
                self._predict_single_regression(knn_labels[i], knn_w[i])
                for i in range(X_test.shape[0])
            ])

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test = np.asarray(y_test)
        if self.task == "classification":
            return np.mean(y_pred == y_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1.0 - ss_res / ss_tot   # R² score