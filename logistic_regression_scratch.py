import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, a, y):
        m = len(y)
        return -(1/m) * np.sum(
            y * np.log(a + 1e-9) + (1 - y) * np.log(1 - a + 1e-9)
        )

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
        
        m = len(y)
        for _ in range(self.n_iterations):
            z = np.dot(X, self.theta) + self.bias
            a = self.sigmoid(z)
            cost = self.compute_cost(a, y)
            self.cost_history.append(cost)
            dz = a - y
            dtheta = (1 / m) * np.dot(X.T, dz)
            dbias = (1 / m) * np.sum(dz)
            self.theta -= self.lr * dtheta
            self.bias -= self.lr * dbias

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.theta) + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)