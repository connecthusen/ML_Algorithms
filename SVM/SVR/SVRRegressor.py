import numpy as np


class SVRRegressor:
    """
    Support Vector Regression with SMO optimization.
    Supports four kernels: 'linear', 'rbf', 'poly', 'sigmoid'.

    Parameters
    ----------
    C          : float, default=1.0   — regularisation strength (larger = less margin)
    epsilon    : float, default=0.1   — epsilon-tube half-width (insensitive zone)
    kernel     : str,   default='rbf' — 'linear' | 'rbf' | 'poly' | 'sigmoid'
    gamma      : float or 'scale', default='scale' — kernel coefficient for rbf/poly/sigmoid
    degree     : int,   default=3     — degree for poly kernel
    coef0      : float, default=0.0   — free term for poly/sigmoid kernels
    max_iter   : int,   default=1000  — maximum SMO passes
    tol        : float, default=1e-3  — convergence tolerance

    Attributes
    ----------
    coef_         : ndarray (n_features,) — weight vector (linear kernel only)
    intercept_    : float                 — bias term b
    support_      : ndarray               — indices of support vectors
    dual_coef_    : ndarray               — alpha_i - alpha_i* for support vectors
    n_support_    : int                   — number of support vectors
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf',
                 gamma='scale', degree=3, coef0=0.0,
                 max_iter=1000, tol=1e-3):
        self.C        = C
        self.epsilon  = epsilon
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

        self._X_train   = None   # stored for kernel evaluation at predict time
        self._alphas    = None   # alpha_i - alpha_i* for all training points

    def fit(self, X_train, y_train):
        """
        Input:
            X_train : (n_samples, n_features)
            y_train : (n_samples,)
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X_train and y_train sample count mismatch.")

        m = X.shape[0]

        # set gamma if 'scale'
        self._gamma = (1.0 / (X.shape[1] * X.var())) if self.gamma == 'scale' else float(self.gamma)

        # compute full kernel matrix
        K = self._kernel_matrix(X, X)

        # run SMO to get dual variables
        alphas, alphas_star = self._smo(K, y, m)

        # net dual coefficients: alpha_i - alpha_i*
        self._alphas  = alphas - alphas_star
        self._X_train = X

        # support vectors — non-zero dual coefficients
        sv_mask          = np.abs(self._alphas) > 1e-5
        self.support_    = np.where(sv_mask)[0]
        self.dual_coef_  = self._alphas[sv_mask]
        self.n_support_  = len(self.support_)

        # compute bias from support vectors on the margin boundary
        self.intercept_ = self._compute_bias(K, y, alphas, alphas_star, m)

        # weight vector for linear kernel
        if self.kernel == 'linear':
            self.coef_ = self._alphas @ X
        else:
            self.coef_ = None

        return self

    def predict(self, X_test):
        """
        Input  : X_test (n_samples, n_features)
        Output : y_pred (n_samples,)
        """
        if self._alphas is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X_test, dtype=np.float64)
        K = self._kernel_matrix(X, self._X_train)   # (n_test, n_train)
        return K @ self._alphas + self.intercept_   # ŷ = K·alpha + b

    def score(self, X_test, y_test):
        """R² score — how well the model explains variance in y."""
        y      = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)   # R² = 1 - SS_res/SS_tot

    def __repr__(self):
        if self._alphas is None:
            return (f"SVRRegressor(C={self.C}, epsilon={self.epsilon}, "
                    f"kernel={self.kernel!r}, gamma={self.gamma})")
        return (f"SVRRegressor(\n"
                f"  kernel={self.kernel!r}, C={self.C}, epsilon={self.epsilon},\n"
                f"  n_support_={self.n_support_}, intercept_={self.intercept_:.4f}\n"
                f")")

    # ── kernels ───────────────────────────────────────────────────────────────

    def _kernel_matrix(self, X1, X2):
        # dispatch to correct kernel — returns (n1, n2) matrix
        if self.kernel == 'linear':
            return X1 @ X2.T

        if self.kernel == 'rbf':
            # K(x,z) = exp(-gamma * ||x - z||^2)
            sq = (np.sum(X1**2, axis=1, keepdims=True)
                  - 2 * X1 @ X2.T
                  + np.sum(X2**2, axis=1))
            return np.exp(-self._gamma * sq)

        if self.kernel == 'poly':
            # K(x,z) = (gamma * x^T z + coef0)^degree
            return (self._gamma * X1 @ X2.T + self.coef0) ** self.degree

        if self.kernel == 'sigmoid':
            # K(x,z) = tanh(gamma * x^T z + coef0)
            return np.tanh(self._gamma * X1 @ X2.T + self.coef0)

        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    # ── SMO optimisation ──────────────────────────────────────────────────────

    def _smo(self, K, y, m):
        """
        Sequential Minimal Optimization for SVR.

        Dual problem — minimise:
            (1/2) * sum_ij (alpha_i - alpha_i*)(alpha_j - alpha_j*) K_ij
            + epsilon * sum_i (alpha_i + alpha_i*)
            - sum_i y_i (alpha_i - alpha_i*)

        Subject to:
            0 <= alpha_i, alpha_i* <= C
            sum_i (alpha_i - alpha_i*) = 0

        Each SMO pass selects pairs of variables and updates them analytically.
        """
        C       = self.C
        eps     = self.epsilon
        tol     = self.tol

        alphas      = np.zeros(m)        # alpha_i
        alphas_star = np.zeros(m)        # alpha_i*
        b           = 0.0

        for iteration in range(self.max_iter):
            n_changed = 0

            for i in range(m):
                # prediction error for sample i
                f_i = float(K[i] @ (alphas - alphas_star)) + b
                r_i = f_i - y[i]

                # KKT violation check
                kkt_violated = (
                    (r_i < -eps - tol and alphas[i] < C) or
                    (r_i > eps + tol  and alphas_star[i] < C) or
                    (r_i < -eps - tol and alphas_star[i] > 0) or
                    (r_i > eps + tol  and alphas[i] > 0)
                )

                if not kkt_violated:
                    continue

                # pick j != i randomly
                j = np.random.randint(0, m)
                while j == i:
                    j = np.random.randint(0, m)

                f_j = float(K[j] @ (alphas - alphas_star)) + b
                r_j = f_j - y[j]

                # save old values
                ai_old  = alphas[i];      ai_star_old  = alphas_star[i]
                aj_old  = alphas[j];      aj_star_old  = alphas_star[j]

                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue

                # update alpha_i and alpha_i* for the epsilon-insensitive loss
                d_alpha = (r_i - r_j) / eta

                alphas[i]      = np.clip(ai_old - d_alpha, 0, C)
                alphas_star[i] = np.clip(ai_star_old + d_alpha, 0, C)

                # enforce balance constraint via alpha_j
                delta = (alphas[i] - ai_old) - (alphas_star[i] - ai_star_old)
                alphas[j]      = np.clip(aj_old - delta, 0, C)
                alphas_star[j] = np.clip(aj_star_old + delta, 0, C)

                # update bias
                db_i = -(r_i + (alphas[i] - ai_old - alphas_star[i] + ai_star_old) * K[i, i])
                db_j = -(r_j + (alphas[j] - aj_old - alphas_star[j] + aj_star_old) * K[j, j])
                b   += 0.5 * (db_i + db_j)

                # check if anything changed
                if (abs(alphas[i] - ai_old) > tol or
                        abs(alphas_star[i] - ai_star_old) > tol):
                    n_changed += 1

            if n_changed == 0:
                break   # converged

        return alphas, alphas_star

    def _compute_bias(self, K, y, alphas, alphas_star, m):
        """Recover b from support vectors on the margin boundary."""
        net    = alphas - alphas_star
        f      = K @ net                       # predictions without bias

        bias_vals = []

        for i in range(m):
            if 0 < alphas[i] < self.C:
                bias_vals.append(y[i] - f[i] - self.epsilon)
            if 0 < alphas_star[i] < self.C:
                bias_vals.append(y[i] - f[i] + self.epsilon)

        if bias_vals:
            return float(np.mean(bias_vals))

        # fallback — use all support vectors
        sv = np.abs(net) > 1e-5
        if sv.any():
            return float(np.mean(y[sv] - f[sv]))

        return 0.0

    def _check_is_fitted(self):
        """Raise RuntimeError if fit() has not been called yet."""
        if self._alphas is None:
            raise RuntimeError("Call fit() before using predict() or score().")
