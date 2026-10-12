"""Check the from-scratch NumPy implementations (Logistic Regression, KNN, linear SVM) against scikit-learn.

The notebook trains these classes but only checks them on one example; this script evaluates them on the
full test set and measures how often they agree with the scikit-learn models.

Run from the repository root:  python src/scratch_vs_sklearn.py
"""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES, RANDOM_STATE, TARGET, load_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ScratchLogisticRegression:
    def __init__(self, lr=0.01, n_itrs=5000):
        self.lr, self.n_itrs = lr, n_itrs
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0
        for _ in range(self.n_itrs):  # gradient descent
            predictions = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.array([0 if p <= 0.5 else 1 for p in y_pred])


class ScratchKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        labels = [self.y_train[i] for i in k_indices]
        return Counter(labels).most_common()[0][0]


class ScratchSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_itrs=5000):
        self.lr, self.lambda_param, self.n_itrs = lr, lambda_param, n_itrs
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.random.rand(n_features)
        self.bias = 0
        for _ in range(self.n_itrs):
            for idx, xi in enumerate(X):
                condition = y_[idx] * (np.dot(xi, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(xi, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        return np.where(np.sign(np.dot(X, self.weights) - self.bias) == -1, 0, 1)  # map {-1, 1} -> {0, 1}


def compare(X_train, y_train, X_test, y_test, seed=RANDOM_STATE):
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    pairs = [
        ("Logistic Regression", ScratchLogisticRegression(), SkLogisticRegression(random_state=seed)),
        ("KNN (k=5)", ScratchKNN(), KNeighborsClassifier(n_neighbors=5)),
        ("SVM (linear)", ScratchSVM(), SVC(kernel="linear", random_state=seed)),
    ]
    rows = []
    for name, scratch, sk in pairs:
        np.random.seed(seed)
        scratch.fit(X_train_s, y_train)
        sk.fit(X_train_s, y_train)
        p_scratch, p_sk = scratch.predict(X_test_s), sk.predict(X_test_s)
        rows.append((name, (p_scratch == y_test).mean(), (p_sk == y_test).mean(), (p_scratch == p_sk).mean()))
    return rows


def main():
    df = load_data()
    X, y = df[FEATURES].values, df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"{'Model':22s} {'scratch acc':>12s} {'sklearn acc':>12s} {'agreement':>10s}")
    for name, a_scratch, a_sk, agree in compare(X_train, y_train, X_test, y_test):
        print(f"{name:22s} {a_scratch:12.2f} {a_sk:12.2f} {agree:10.2f}")


if __name__ == "__main__":
    main()
