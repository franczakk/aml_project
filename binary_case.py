import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score


class LogRegCCD:
    RECALL = "recall"
    PRECISION = "precision"
    F1 = "f1"
    BALANCED_ACCURACY = "balanced_accuracy"
    ROC_AUC = "roc_auc"
    AVERAGE_PRECISION = "average_precision"

    def __init__(self):
        self.mean = None
        self.std = None
        self.beta = None

    def _prob(self, x):
        """
        counts the probability of y=1 given x
        :param x: vector
        :return: probability value
        """
        product = self.beta @ x
        if product > 0:
            e = np.exp(-product)
            return 1 / (1 + e)
        e = np.exp(product)
        return e / (1 + e)

    @staticmethod
    def _count_weight(p_x):
        """
        computes the weight for beta update
        :param p_x: probability of y=1 given x
        :return: weight of the observation x
        """
        return p_x * (1 - p_x)

    @staticmethod
    def _compute_z(y, x_b, p_x, w_x):
        """
        computes the z value for beta update
        :param y: class 1 indicator
        :param x_b: product of vectors x and beta
        :param p_x: probability of y=1 given x
        :param w_x: weight of observation x
        :return: z value
        """
        z = x_b + (y - p_x) / w_x
        return z

    @staticmethod
    def _soft_thresholding(x, gamma):
        """
        applies soft thresholding to x
        :param x: vector
        :param gamma: threshold
        :return: x after soft thresholding applied
        """
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if x > gamma:
            return x - gamma
        if x < -gamma:
            return x + gamma
        return 0

    def _update(self, X, y, j, alpha, lmbda):
        """
        updates the jth beta parameter
        :param X: training data matrix
        :param y: vector of indicators of class 1
        :param j: beta vector coordinate
        :param alpha: elastic net parameter
        :param lmbda: regularization penalty parameter
        :return: the new value of beta's jth coordinate
        """
        x_j = X[:, j]
        n = len(x_j)
        total = 0
        weights_vector = []
        for i in range(n):
            x_b = X[i] @ self.beta
            prob_x = self._prob(X[i])
            w_x = self._count_weight(prob_x)
            weights_vector.append(w_x)

            x_ij = x_j[i]

            if w_x == 0:
                total += x_ij * (y[i] - prob_x)
                continue

            z_i = self._compute_z(y[i], x_b, prob_x, w_x)

            x_b_j = x_b - x_ij * self.beta[j]
            total += w_x * x_ij * (z_i - x_b_j)

        st = self._soft_thresholding(total, alpha * lmbda)
        denominator = np.array(weights_vector) @ (x_j ** 2) + lmbda * (1 - alpha)

        return st / denominator

    def _coordinate_descent(self, X, y, alpha, lmbda):
        """
        runs one coordinate descent iteration
        :param X: training data matrix
        :param y: vector of indicators of class 1
        :param alpha: elastic net parameter
        :param lmbda: regularization penalty parameter
        :return: None
        """
        for j in range(len(self.beta)):
            self.beta[j] = self._update(X, y, j, alpha, lmbda)

    def compute_lambda_max(self, X, y, alpha, center=True):
        """
        computes the maximum value of lambda to consider, assumes that beta is equal to 0
        :param X: training data matrix
        :param y: vector of indicators of class 1
        :param alpha: elastic net parameter
        :return: maximum value of lambda
        """
        if alpha == 0:
            return 0
        # apply on standardized data
        X = self._standardize_matrix(X, center)
        # we assume that beta == 0
        z = 4 * y - 2
        # w = 0.25
        return np.max(np.abs(X.T @ z) * 0.25) / alpha

    def fit(self, iterations, X_train, y_train, alpha, lmbda, beta_eps=0, center=True, adjust_beta=True):
        """
        fits the model to the training data by estimating beta parameter
        warm start is applied
        :param iterations: no. of iterations
        :param X_train: training data matrix
        :param y_train: binary vector of classes
        :param alpha: elastic net parameter
        :param lmbda: regularization penalty parameter
        :param beta_eps: stop condition parameter
        :param center: apply centering to X_train (do not apply for sparse data)
        :param adjust_beta: apply adjustment to beta parameter
        :return: None
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1] interval")
        if lmbda < 0:
            raise ValueError("lmbda cannot be less than 0")
        # TODO iterate only over active coordinates
        # TODO add covariance updates
        # normalize data, add intercept, prepare beta
        X_train = self._initialize(X_train, center)
        for i in range(iterations):
            beta_old = self.beta.copy()
            self._coordinate_descent(X_train, y_train, alpha, lmbda)
            # stop condition
            if np.linalg.norm(self.beta - beta_old) < beta_eps:
                break
        # rescaling betas to the original data
        if adjust_beta:
            self._adjust_beta()

    def validate(self, X_valid, y_valid, measure):
        """
        computes the value of measure on the validation set
        :param X_valid: validation data matrix
        :param y_valid: binary vector of classes
        :param measure: evaluation measure
        :return: evaluation value
        """
        y_prob = self.predict_proba(X_valid)
        y_pred = self.predict(X_valid)
        if measure == LogRegCCD.RECALL:
            return recall_score(y_valid, y_pred)
        if measure == LogRegCCD.PRECISION:
            return precision_score(y_valid, y_pred)
        if measure == LogRegCCD.F1:
            return f1_score(y_valid, y_pred)
        if measure == LogRegCCD.BALANCED_ACCURACY:
            return balanced_accuracy_score(y_valid, y_pred)
        if measure == LogRegCCD.ROC_AUC:
            return roc_auc_score(y_valid, y_prob)
        if measure == LogRegCCD.AVERAGE_PRECISION:
            return average_precision_score(y_valid, y_prob)

    def plot(self, measure, X_train, y_train, X_valid, y_valid, alpha, center=True, eps=1e-3, k=100, iterations=10, beta_eps=0):
        """
        produces plot showing how the given evaluation measure changes with lambda
        :param measure: evaluation measure
        :param X_train: training data matrix
        :param y_train: binary vector of classes
        :param X_valid: validation data matrix
        :param y_valid: binary vector of classes
        :param alpha: elastic net parameter
        :param center: apply centering to X_train (do not apply for sparse data)
        :param eps: defines the minimal value of lambda (eps * lambda_max)
        :param k: number of lambda values
        :param iterations: number of iterations for each lambda value
        :param beta_eps: stop condition parameter
        :return: None
        """
        # computing lambda max
        lambda_max = self.compute_lambda_max(X_train, y_train, alpha, center)
        lambda_min = lambda_max * eps
        # constructing sequence of lambda values on log scale
        lambda_values = np.logspace(np.log10(lambda_max), np.log10(lambda_min), k)
        evaluation_values = []
        for lmbda in lambda_values:
            # fitting model
            self.fit(iterations, X_train, y_train, alpha, lmbda, beta_eps=beta_eps, center=center)
            # evaluation
            eval = self.validate(X_valid, y_valid, measure)
            evaluation_values.append(eval)
            self._scale_beta_back()
        # showing figure
        plt.semilogx(lambda_values, evaluation_values)
        plt.xlabel("lambda")
        plt.ylabel("evaluation score")
        plt.title(f"{measure} for lambda values")
        plt.show()

    def plot_coefficients(self, X_train, y_train, alpha, center=True, eps=1e-3, k=100, iterations=10, beta_eps=0):
        """
        produces plot showing the coefficient values as function of lambda parameter
        :param X_train: training data matrix
        :param y_train: binary vector of classes
        :param alpha: elastic net parameter
        :param center: apply centering to X_train (do not apply for sparse data)
        :param eps: defines the minimal value of lambda (eps * lambda_max)
        :param k: number of lambda values
        :param iterations: number of iterations for each lambda value
        :param beta_eps: stop condition parameter
        :return: None
        """
        # computing lambda max
        lambda_max = self.compute_lambda_max(X_train, y_train, alpha, center)
        lambda_min = lambda_max * eps
        # constructing sequence of lambda values on log scale
        lambda_values = np.logspace(np.log10(lambda_max), np.log10(lambda_min), k)
        coefficient_values = []
        for lmbda in lambda_values:
            # fitting model
            self.fit(iterations, X_train, y_train, alpha, lmbda, beta_eps=beta_eps, center=center)
            coefficient_values.append(self.beta.copy())
            self._scale_beta_back()
        # showing figure
        plt.figure(figsize=(10, 6))
        plt.semilogx(lambda_values, coefficient_values)
        plt.xlabel("lambda")
        plt.ylabel("coefficients")
        plt.title("Estimated coefficients for lambda values")
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.show()

    def _standardize_matrix(self, X, center=True):
        """
        computes mean and standard deviation for each feature and standardizes data
        :param X: training data matrix
        :param center: apply centering to X_train (do not apply for sparse data)
        :return: standardized matrix
        """
        if center:
            self.mean = np.mean(X, axis=0)
        else:
            self.mean = np.zeros(X.shape[1])
        self.std = np.std(X, axis=0)
        X_standardized = (X - self.mean) / self.std
        return X_standardized

    @staticmethod
    def _add_intercept(X):
        """
        adds intercept to data
        :param X: training data matrix
        :return: data with intercept
        """
        X_1 = np.c_[np.ones(X.shape[0]), X]
        return X_1

    def _init_beta(self, n_features):
        """
        initializes the beta parameter with 0 values
        :param n_features: number of features
        :return: initial beta parameter
        """
        self.beta = np.zeros(n_features)

    def _initialize(self, X, center=True):
        """
        prepares data for processing, applies standardizing and adds intercept, initializes beta parameter
        :param X: training data matrix
        :param center: apply centering to X_train (do not apply for sparse data)
        :return: preprocessed data
        """
        X = self._standardize_matrix(X, center)
        X = self._add_intercept(X)
        n_features = X.shape[1]
        if self.beta is None:  # do not reset beta
            self._init_beta(n_features)
        return X

    def _adjust_beta(self):
        """
        modifies the beta parameter to original data
        :return: modified beta parameter
        """
        # rescaling beta vector to original data
        self.beta[1:] /= self.std
        # intercept - applying new beta
        self.beta[0] -= np.sum(self.mean * self.beta[1:])

    def _scale_beta_back(self):
        """
        scales the beta parameter to standardized data
        :return: modified beta parameter
        """
        self.beta[0] += np.sum(self.mean * self.beta[1:])
        self.beta[1:] *= self.std

    def predict_proba(self, X_test):
        """
        computes the predicted probabilities of class 1
        :param X_test: test data matrix
        :return: vector of predicted probabilities
        """
        X_test = self._add_intercept(X_test)
        n = len(X_test)
        probas = np.zeros(n)
        for i in range(n):
            probas[i] = self._prob(X_test[i])
        return probas

    def predict(self, X_test):
        """
        predicts the class for given data
        :param X_test: test data matrix
        :return: vector of predicted classes
        """
        probas = self.predict_proba(X_test)
        return (probas >= 0.5).astype(int)
