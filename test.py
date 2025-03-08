import numpy as np

global betas

# global exp_sum_nominators
# global exp_sum_d

def product(x, beta):
    return x @ beta

def p(x, beta):
    exp_sum = np.sum([np.exp(b @ x) for b in betas])
    return np.exp(beta @ x) / (1 + exp_sum)

def weight(p_x):
    # p_x needs x_i and beta
    return p_x * (1 - p_x)

# compute_z
def compute_z(y, x_b, p_x, w_x):
    # y == indicator
    z = x_b + (y - p_x) / w_x
    return z

def soft_thresholding(x, gamma):
    # gamma > 0
    if x > gamma:
        return x - gamma
    if x < -gamma:
        return x + gamma
    return 0

def dummy_update(X, y, j, beta, alpha, l):
    x_j = X[:, j]
    # x_j - jth column
    n = len(x_j) # no. of observations
    sum = 0
    weights_vector = []
    for i in range(n):
        x_b = X[i] @ beta
        prob_x = p(X[i], beta)
        w_x = weight(prob_x)
        weights_vector.append(w_x)
        x_ij = x_j[i]

        z_i = compute_z(y[i], x_b, prob_x, w_x)

        # x_b <- x_b_i
        # sum += w_x * x_ij * (z_i - p_x + x_ij * beta[j])
        sum += w_x * x_ij * (z_i - x_b)

    st = soft_thresholding(sum, alpha * l)
    denominator = np.array(weights_vector) @ (x_j ** 2) + l * (1 - alpha)

    return st / denominator

def coordinate_descent(X, y, k, alpha, l):
    beta = betas[k]
    for j in range(len(beta)):
        # old_beta_j = beta[j]
        beta[j] = dummy_update(X, y, j, beta, alpha, l)

        # change = -np.exp(old_beta_j) + np.exp(beta[j])
        # exp_sum_n += change
        # exp_sum_d += change
    return beta

def train(iterations, X, y, alpha, l):
    for i in range(iterations):
        for k in range(len(betas)):
            # iterating over classes
            # y == 1 for jth class, 0 for the rest
            # levels from 0 to K-1, Kth is the reference level
            y_k = np.where(y == k, 1, 0) # may be prepared once and stored in one matrix
            betas[k] = coordinate_descent(X, y_k, k, alpha, l)

def standardize_matrix(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

def add_intercept(X):
    X_1 = np.c_[np.ones(X.shape[0]), X]
    return X_1

def init_beta(n_features):
    return np.ones(n_features)

def initialize(X, n_classes):
    # assumes that y values are 0, 1, ..., n_classes - 1
    X, mean, std = standardize_matrix(X)
    X = add_intercept(X)
    n_features = X.shape[1]
    betas = [init_beta(n_features) for _ in range(n_classes - 1)] # 1 reference level
    # save mean, std

def adjust_betas(mean, std):
    for i in range(len(betas)):
        beta = betas[i]
        beta[0] += np.sum(mean) # vector of means (+/-)
        beta[1:] *= std # dot product

def predict_proba(X_test):
    n_classes = len(betas) + 1
    probas = np.zeros((len(X_test), n_classes))
    for i in range(len(X_test)):
        # count exp_sum once
        for k in range(len(betas)):
            probas[i, k] = p(X_test[i], betas[k])
    # fill the reference level
    probas[:, -1] = 1 - np.sum(probas[:, :-1], axis=1)
    return probas

def predict(X_test):
    probas = predict_proba(X_test)
    return np.argmax(probas, axis=1)

#%%

n_features = 2
n_classes = 2

betas = [np.random.rand(n_features) for _ in range(n_classes - 1)]

def shuffle(X, y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def generate_scheme_1(n, p, a):
    y = np.random.randint(0, 2, n)
    n_0 = len(y[y == 0])
    X_0 = np.random.normal(loc=0, scale=1, size=(n_0, p))
    X_1 = np.random.normal(loc=a, scale=1, size=(n - n_0, p))
    X = np.concatenate((X_0, X_1))
    y = np.concatenate((y[y == 0], y[y == 1]))
    X, y = shuffle(X, y)
    return X, y

X, y = generate_scheme_1(100, 2, 5)

train(1000, X, y, 0.5, 0.5)

y_pred = predict(X)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0])
plt.scatter(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1])
plt.show()


