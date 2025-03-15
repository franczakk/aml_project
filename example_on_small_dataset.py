import numpy as np
from logistic_regression import LogRegCCD
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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


n = 100
n_features = 2
n_classes = 2
alpha = 0.9

X, y = generate_scheme_1(n, n_features, 5)

lr = LogRegCCD()
lr.fit(10, X, y, alpha, 0.9)
print(lr.beta)
y_pred = lr.predict(X)

accuracy = accuracy_score(y, y_pred)
print(accuracy)

plt.scatter(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0])
plt.scatter(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1])
plt.show()

#%% highest lambda value

lr = LogRegCCD()
# computing lambda max -> beta = 0
lambda_max = lr.compute_lambda_max(X, y, alpha)
lr.fit(1000, X, y, alpha, lambda_max)
print(lr.beta)
y_pred = lr.predict(X)

accuracy = accuracy_score(y, y_pred)
print(accuracy)

plt.scatter(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0])
plt.scatter(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1])
plt.show()

#%%

alpha = 0.9
lr = LogRegCCD()
lr.plot(LogRegCCD.F1, X, y, X, y, 0.9)

#%%

alpha = 0.9
lr = LogRegCCD()
lr.plot_coefficients(X, y, alpha)
