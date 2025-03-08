import numpy as np


class LogRegCCD:
    def __init__(self):
        self.mean = None
        self.std = None
        self.beta = None

    def prob(self, x):
        product = np.exp(self.beta @ -x)
        return 1 / (1 + product)

    def count_weight(self, p_x):
        return p_x * (1 - p_x)

    def compute_z(self, y, x_b, p_x, w_x):
        # y == indicator
        # eps = 1e-10
        # if w_x == 0:
        #     w_x = eps
        z = x_b + (y - p_x) / w_x
        return z

    def soft_thresholding(self, x, gamma):
        # gamma > 0
        if x > gamma:
            return x - gamma
        if x < -gamma:
            return x + gamma
        return 0

    def dummy_update(self, X, y, j, alpha, l):
        # update self.beta[j]
        # j-th feature
        x_j = X[:, j]
        # x_j - jth column
        n = len(x_j)  # no. of observations
        sum = 0
        weights_vector = []
        for i in range(n):
            x_b = X[i] @ self.beta
            prob_x = self.prob(X[i])
            w_x = self.count_weight(prob_x)
            weights_vector.append(w_x)
            x_ij = x_j[i]
            # y[i] == indicator
            z_i = self.compute_z(y[i], x_b, prob_x, w_x)

            # sum += w_x * x_ij * (z_i - p_x + x_ij * beta[j])
            x_b_j = x_b - x_ij * self.beta[j]
            sum += w_x * x_ij * (z_i - x_b_j)

        st = self.soft_thresholding(sum, alpha * l)
        denominator = np.array(weights_vector) @ (x_j ** 2) + l * (1 - alpha)

        return st / denominator

    def coordinate_descent(self, X, y, alpha, l):
        for j in range(len(self.beta)):
            self.beta[j] = self.dummy_update(X, y, j, alpha, l)

    def fit(self, iterations, X, y, alpha, l):
        X = self.initialize(X)
        for i in range(iterations):
            # y - binary vector
            self.coordinate_descent(X, y, alpha, l)

    def standardize_matrix(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_standardized = (X - self.mean) / self.std
        return X_standardized

    def add_intercept(self, X):
        X_1 = np.c_[np.ones(X.shape[0]), X]
        return X_1
    
    def prepare_matrix(self, X):
        X = (X - self.mean) / self.std
        X = self.add_intercept(X)
        return X

    def init_beta(self, n_features):
        self.beta = np.ones(n_features)
        # self.beta[0] = 1
        
    def initialize(self, X):
        X = self.standardize_matrix(X)
        X = self.add_intercept(X)
        n_features = X.shape[1]
        if self.beta is None:
            self.init_beta(n_features)
        return X

    def adjust_betas(self):
        self.beta[0] += np.sum(self.mean)
        self.beta[1:] *= self.std

    def predict_proba(self, X_test):
        X_test = self.prepare_matrix(X_test)
        n = len(X_test)
        probas = np.zeros(n)
        for i in range(n):
            probas[i] = self.prob(X_test[i])
        return probas  # probabilities of class 1

    def predict(self, X_test):
        probas = self.predict_proba(X_test)
        return (probas >= 0.5).astype(int)
