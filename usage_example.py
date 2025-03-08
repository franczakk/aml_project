import numpy as np
from binary_case import LogRegCCD
import matplotlib.pyplot as plt


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

X, y = generate_scheme_1(n, n_features, 5)

lr = LogRegCCD()
lr.fit(1000, X, y, 0.9, 0.9)
lr.beta
y_pred = lr.predict(X)

plt.scatter(X[:, 0][y_pred == 0], X[:, 1][y_pred == 0])
plt.scatter(X[:, 0][y_pred == 1], X[:, 1][y_pred == 1])
plt.show()
