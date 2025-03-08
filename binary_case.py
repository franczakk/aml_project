import numpy as np


class LogRegCCD:
    def __init__(self):
        self.mean = None
        self.std = None
        self.beta = None

    def _prob(self, x):
        product = self.beta @ x
        if product > 0:
            e = np.exp(-product)
            return 1 / (1 + e)
        e = np.exp(product)
        return e / (1 + e)

    @staticmethod
    def _count_weight(p_x):
        return p_x * (1 - p_x)

    @staticmethod
    def _compute_z(y, x_b, p_x, w_x):
        # y == indicator
        z = x_b + (y - p_x) / w_x
        return z

    @staticmethod
    def _soft_thresholding(x, gamma):
        # gamma > 0
        if x > gamma:
            return x - gamma
        if x < -gamma:
            return x + gamma
        return 0

    def _update(self, X, y, j, alpha, lmbda):
        # update self.beta[j]
        # j-th feature
        x_j = X[:, j]
        # x_j - jth column
        n = len(x_j)  # no. of observations
        total = 0
        weights_vector = []
        for i in range(n):
            x_b = X[i] @ self.beta
            prob_x = self._prob(X[i])
            w_x = self._count_weight(prob_x)
            weights_vector.append(w_x)
            x_ij = x_j[i]
            # y[i] == indicator
            z_i = self._compute_z(y[i], x_b, prob_x, w_x)

            x_b_j = x_b - x_ij * self.beta[j]
            total += w_x * x_ij * (z_i - x_b_j)

        st = self._soft_thresholding(total, alpha * lmbda)
        denominator = np.array(weights_vector) @ (x_j ** 2) + lmbda * (1 - alpha)

        return st / denominator

    def _coordinate_descent(self, X, y, alpha, lmbda):
        for j in range(len(self.beta)):
            self.beta[j] = self._update(X, y, j, alpha, lmbda)

    def compute_lambda_max(self, X, y, alpha):
        if alpha == 0:
            return 0
        # apply on standardized data
        # TODO it changes global parameters
        X = self._standardize_matrix(X)
        # N * alpha
        # we assume that beta == 0
        z = 4 * y - 2
        # w = 0.25
        #  lambda_min = eps * lambda_max, eps = 1e-3, K = 100
        # lambda_values = np.logspace(np.log10(lambda_max), np.log10(lambda_min), K)
        return np.max(np.abs(X.T @ z) * 0.25) / alpha

    def fit(self, iterations, X_train, y_train, alpha, lmbda):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1] interval")
        if lmbda < 0:
            raise ValueError("lmbda cannot be less than 0")
        # TODO add stop condition
        # TODO iterate only over active coordinates
        # TODO add covariance updates
        # TODO - in sparse data case do not center it
        # warm start
        X_train = self._initialize(X_train)
        for i in range(iterations):
            # y_train - binary vector
            self._coordinate_descent(X_train, y_train, alpha, lmbda)
        # rescaling betas to original data
        self._adjust_beta()

    def validate(self, X_valid, y_valid, measure):
        # TODO
        pass

    def plot(self, measure):
        # TODO
        pass

    def plot_coefficients(self):
        # TODO
        pass

    def _standardize_matrix(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_standardized = (X - self.mean) / self.std
        return X_standardized

    @staticmethod
    def _add_intercept(X):
        X_1 = np.c_[np.ones(X.shape[0]), X]
        return X_1

    def _init_beta(self, n_features):
        self.beta = np.zeros(n_features)

    def _initialize(self, X):
        X = self._standardize_matrix(X)
        X = self._add_intercept(X)
        n_features = X.shape[1]
        if self.beta is None:  # do not reset beta
            self._init_beta(n_features)
        return X

    def _adjust_beta(self):
        # rescaling beta vector to original data
        self.beta[1:] /= self.std
        # intercept - applying new beta
        self.beta[0] -= np.sum(self.mean * self.beta[1:])

    def predict_proba(self, X_test):
        X_test = self._add_intercept(X_test)
        n = len(X_test)
        probas = np.zeros(n)
        for i in range(n):
            probas[i] = self._prob(X_test[i])
        return probas  # probabilities of class 1

    def predict(self, X_test):
        probas = self.predict_proba(X_test)
        return (probas >= 0.5).astype(int)
