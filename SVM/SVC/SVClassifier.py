import numpy as np


class SVClassifier:
    """
    Support Vector Classification with SMO optimization.
    Multiclass via One-vs-Rest (OvR). Supports 4 kernels.

    Parameters
    ----------
    C          : float, default=1.0   — regularisation (larger = less margin)
    kernel     : str,   default='rbf' — 'linear' | 'rbf' | 'poly' | 'sigmoid'
    gamma      : float or 'scale', default='scale' — kernel coefficient
    degree     : int,   default=3     — degree for poly kernel
    coef0      : float, default=0.0   — free term for poly/sigmoid kernels
    max_iter   : int,   default=1000  — maximum SMO passes per binary classifier
    tol        : float, default=1e-3  — convergence tolerance

    Attributes
    ----------
    coef_        : ndarray (n_classes, n_features) — weights (linear kernel only)
    intercept_   : ndarray (n_classes,)            — bias per binary classifier
    support_     : list of ndarray                 — support vector indices per class
    dual_coef_   : list of ndarray                 — alpha_i for each binary classifier
    n_support_   : ndarray (n_classes,)            — number of SVs per class
    classes_     : ndarray                         — unique class labels
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 degree=3, coef0=0.0, max_iter=1000, tol=1e-3):
        self.C        = C
        self.kernel   = kernel
        self.gamma    = gamma
        self.degree   = degree
        self.coef0    = coef0
        self.max_iter = max_iter
        self.tol      = tol

        self.coef_      = None
        self.intercept_ = None
        self.support_   = None
        self.dual_coef_ = None
        self.n_support_ = None
        self.classes_   = None

        self._alphas    = None   # list of alpha arrays, one per binary classifier
        self._X_train   = None
        self._gamma_val = None

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_train and y_train sample count mismatch.")

        self.classes_   = np.unique(y)
        self._X_train   = X
        self._gamma_val = (1.0 / (X.shape[1] * X.var())) if self.gamma == 'scale' else float(self.gamma)

        K = self._kernel_matrix(X, X)

        # OvR — train one binary SVM per class
        self._alphas    = []
        self.intercept_ = []
        self.support_   = []
        self.dual_coef_ = []
        n_support       = []

        for cls in self.classes_:
            y_bin = np.where(y == cls, 1.0, -1.0)   # +1 = this class, -1 = rest
            alphas, b = self._smo(K, y_bin, X.shape[0])

            self._alphas.append(alphas)
            self.intercept_.append(b)

            sv_mask = np.abs(alphas) > 1e-5
            self.support_.append(np.where(sv_mask)[0])
            self.dual_coef_.append(alphas[sv_mask])
            n_support.append(int(sv_mask.sum()))

        self.intercept_ = np.array(self.intercept_)
        self.n_support_ = np.array(n_support)

        # weight vectors for linear kernel only
        if self.kernel == 'linear':
            self.coef_ = np.array([
                alphas @ X for alphas in self._alphas
            ])
        else:
            self.coef_ = None

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,) — class labels
        """
        if self._alphas is None:
            raise RuntimeError("Call fit() before predict().")

        scores = self.decision_function(X_test)
        if scores.ndim == 1:
            return self.classes_[(scores >= 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    def decision_function(self, X_test):
        """
        Raw decision scores — (n_samples, n_classes) for multiclass.
        Argmax gives the predicted class.
        """
        if self._alphas is None:
            raise RuntimeError("Call fit() before predict().")

        X    = np.asarray(X_test, dtype=np.float64)
        K    = self._kernel_matrix(X, self._X_train)   # (n_test, n_train)
        scores = np.column_stack([
            K @ alphas + b
            for alphas, b in zip(self._alphas, self.intercept_)
        ])
        return scores

    def score(self, X_test, y_test):
        """Accuracy — fraction of correct predictions."""
        return np.mean(self.predict(X_test) == np.asarray(y_test))

    def __repr__(self):
        if self._alphas is None:
            return (f"SVClassifier(C={self.C}, kernel={self.kernel!r}, "
                    f"gamma={self.gamma})")
        return (f"SVClassifier(\n"
                f"  kernel={self.kernel!r}, C={self.C},\n"
                f"  classes_={self.classes_},\n"
                f"  n_support_={self.n_support_},\n"
                f"  intercept_={self.intercept_}\n"
                f")")

    # kernels
    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T

        if self.kernel == 'rbf':
            sq = (np.sum(X1**2, axis=1, keepdims=True)
                  - 2 * X1 @ X2.T
                  + np.sum(X2**2, axis=1))
            return np.exp(-self._gamma_val * sq)

        if self.kernel == 'poly':
            return (self._gamma_val * X1 @ X2.T + self.coef0) ** self.degree

        if self.kernel == 'sigmoid':
            return np.tanh(self._gamma_val * X1 @ X2.T + self.coef0)

        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    # SMO

    def _smo(self, K, y, m):
        """
        SMO for binary SVM classification.

        Dual problem — minimise:
            (1/2) * sum_ij alpha_i alpha_j y_i y_j K_ij - sum_i alpha_i
        Subject to:
            0 <= alpha_i <= C
            sum_i alpha_i y_i = 0

        Updates two alphas analytically per iteration.
        Returns (alphas * y_binary, bias).
        """
        C   = self.C
        tol = self.tol

        alphas = np.zeros(m)
        b      = 0.0

        for iteration in range(self.max_iter):
            n_changed = 0

            for i in range(m):
                # decision value and error
                f_i   = float((alphas * y) @ K[i]) + b
                err_i = f_i - y[i]

                # KKT violation
                kkt = ((err_i * y[i] < -tol and alphas[i] < C) or
                       (err_i * y[i] > tol  and alphas[i] > 0))

                if not kkt:
                    continue

                # pick j != i randomly
                j = np.random.randint(0, m)
                while j == i:
                    j = np.random.randint(0, m)

                f_j   = float((alphas * y) @ K[j]) + b
                err_j = f_j - y[j]

                ai_old = alphas[i]
                aj_old = alphas[j]

                # compute bounds L, H for alpha_j
                if y[i] == y[j]:
                    L = max(0, ai_old + aj_old - C)
                    H = min(C, ai_old + aj_old)
                else:
                    L = max(0, aj_old - ai_old)
                    H = min(C, C + aj_old - ai_old)

                if L >= H:
                    continue

                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue

                # update alpha_j
                alphas[j] = np.clip(aj_old - y[j] * (err_i - err_j) / eta, L, H)

                if abs(alphas[j] - aj_old) < 1e-5:
                    continue

                # update alpha_i
                alphas[i] = ai_old + y[i] * y[j] * (aj_old - alphas[j])

                # update bias
                b1 = b - err_i - y[i]*(alphas[i]-ai_old)*K[i,i] - y[j]*(alphas[j]-aj_old)*K[i,j]
                b2 = b - err_j - y[i]*(alphas[i]-ai_old)*K[i,j] - y[j]*(alphas[j]-aj_old)*K[j,j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                n_changed += 1

            if n_changed == 0:
                break

        return alphas * y, b   # signed dual coefficients

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self._alphas is None:
            raise RuntimeError("Call fit() before using predict() or score().")
