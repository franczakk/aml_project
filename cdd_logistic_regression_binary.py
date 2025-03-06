import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Computes the sigmoid function for input x
    return 1 / (1 + np.exp(-x))

def p(x, beta):
    # Computes the probability of the class being 1 given input x and model parameters beta
    return sigmoid(x @ beta)

def weight(p_x):
    # Computes the weight for a given probability p_x using the formula p_x * (1 - p_x)
    return p_x * (1 - p_x)

def compute_z(y_i, x_b, p_x, w_x):
    # Computes the z value used in the update of the coefficients during the coordinate descent
    return x_b + (y_i - p_x) / w_x

def soft_thresholding(x, gamma):
    # Applies soft thresholding to the value x with threshold gamma, 
    # reducing |x| > gamma, or making them 0 otherwise
    if x > gamma:
        return x - gamma
    if x < -gamma:
        return x + gamma
    return 0

def dummy_update(X, y, j, beta, alpha, l):
    # Updates the j-th coefficient beta[j] based on the coordinate descent algorithm
    # l - regularisation strength
    x_j = X[:, j]
    n = len(x_j)
    sum_val = 0
    weights_vector = []
    for i in range(n):
        x_b = X[i] @ beta
        prob_x = p(X[i], beta)
        w_x = weight(prob_x)
        weights_vector.append(w_x) # holds the weights corresponding to each data point, calculated based on the probability of the data point belonging to class 1 (p_x).
        z_i = compute_z(y[i], x_b, prob_x, w_x) # weighted residual between the actual label y_i and the predicted probability p_x
        sum_val += w_x * x_j[i] * (z_i - x_b) # accumulates the weighted gradient contributions from all data points
    
    soft_thresh = soft_thresholding(sum_val, alpha * l) # enforcing Lasso regularization by pushing small coefficients toward zero
    denominator = np.array(weights_vector) @ (x_j ** 2) + l * (1 - alpha) # alpha = 1 -> only lasso; sum of squared feature values, weighted by the inverse of the coefficients
    return soft_thresh / denominator

def coordinate_descent(X, y, alpha, l, beta):
    # Performs coordinate descent optimization on the model parameters beta
    for j in range(len(beta)):
        beta[j] = dummy_update(X, y, j, beta, alpha, l)
    return beta

def train(iterations, X, y, alpha, l, beta):
    # Trains the model using coordinate descent for a specified number of iterations
    for _ in range(iterations):
        beta = coordinate_descent(X, y, alpha, l, beta)
    return beta

def predict_proba(X, beta):
    # Predicts the probabilities for the input X based on the model parameters beta
    return p(X, beta)

def predict(X, beta):
    # Predicts the binary class labels (0 or 1) based on the predicted probabilities
    return (predict_proba(X, beta) >= 0.5).astype(int)

def standardize_matrix(X):
    # Standardizes the input matrix X by subtracting the mean and dividing by the standard deviation
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def add_intercept(X):
    # Adds a column of ones to the input matrix X for the intercept term in the linear model
    return np.c_[np.ones(X.shape[0]), X]

def shuffle(X, y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]

def generate_data(n, p, a):
    # Generates synthetic data with n samples, p features, and class separation parameter a
    y = np.random.randint(0, 2, n)
    X_0 = np.random.normal(loc=0, scale=1, size=(sum(y == 0), p))
    X_1 = np.random.normal(loc=a, scale=1, size=(sum(y == 1), p))
    X = np.vstack((X_0, X_1))
    y = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))
    return shuffle(X, y)


n_features = 2
X, y = generate_data(100, n_features, 5)
X, mean, std = standardize_matrix(X)
X = add_intercept(X)
beta = np.random.rand(X.shape[1])


beta = train(1000, X, y, 0.5, 0.5, beta)
y_pred = predict(X, beta)


plt.scatter(X[:, 1][y_pred == 0], X[:, 2][y_pred == 0], label="Class 0")
plt.scatter(X[:, 1][y_pred == 1], X[:, 2][y_pred == 1], label="Class 1")
plt.legend()
plt.show()
