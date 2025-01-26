import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_scratch import LogisticRegressionScratch


X = pd.read_csv("logisticX.csv")
y = pd.read_csv("logisticY.csv")

X = X.values
y = y.values.ravel()

model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
final_cost = model.cost_history[-1]
print("Final Cost:", final_cost)
print("Coefficients:", model.theta)
print("Bias:", model.bias)

# Plot cost vs iteration for the first 50 iterations
plt.figure()
plt.plot(model.cost_history[:50])
plt.title("Cost vs Iteration (first 50)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Plot data points and decision boundary
plt.figure()
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Class 1')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5], colors='green')

plt.legend()
plt.title("Data points and decision boundary")
plt.show()

# Create new features X^2
X_extended = np.hstack([X, X[:, 0:2]**2])  # assume first two columns are the original features
model2 = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model2.fit(X_extended, y)

# Plot new boundary
plt.figure()
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Class 1')

x_min2, x_max2 = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min2, y_max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.linspace(x_min2, x_max2, 100),
                       np.linspace(y_min2, y_max2, 100))

# Extend the grid with squared features
xx2_sq = xx2**2
yy2_sq = yy2**2
grid_extended = np.c_[xx2.ravel(), yy2.ravel(), xx2_sq.ravel(), yy2_sq.ravel()]

Z2 = model2.predict(grid_extended)
Z2 = Z2.reshape(xx2.shape)
plt.contour(xx2, yy2, Z2, levels=[0.5], colors='purple')

plt.legend()
plt.title("Extended data points and decision boundary")
plt.show()

# Confusion matrix
y_pred = model.predict(X)
tp = np.sum((y_pred == 1) & (y == 1))
tn = np.sum((y_pred == 0) & (y == 0))
fp = np.sum((y_pred == 1) & (y == 0))
fn = np.sum((y_pred == 0) & (y == 1))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

print("Confusion Matrix:")
print([[tn, fp],[fn, tp]])
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)