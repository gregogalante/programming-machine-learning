import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mnist as data

"""
The sigmoid method will be used to calculate the sigmoid of a given value.
"""
def sigmoid(z):
  return 1/(1 + np.exp(-z))

"""
The forward method will be used to calculate the forward pass of the model.
"""
def forward(X, w):
  weighted_sum = np.matmul(X, w)
  return sigmoid(weighted_sum)

"""
The classify method will be used to classify the data based on the forward pass.
"""
def classify(X, w):
  return np.round(forward(X, w))

"""
The loss method will be used to calculate the loss of the model.
"""
def loss(X, Y, w):
  y_hat = forward(X, w)
  first_term = Y * np.log(y_hat)
  second_term = (1 - Y) * np.log(1 - y_hat)
  return -np.average(first_term + second_term)

"""
The gradient method will be used to calculate the gradient of the model.
"""
def gradient(X, Y, w):
  return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

"""
The train method will be used to train the model.
"""
def train(X, Y, learning_rate=0.001, epochs=100000):
  w = np.zeros((X.shape[1], 1))
  for i in range(epochs):
    w -= learning_rate * gradient(X, Y, w)
    if i % 10 == 0:
      print(f'Iteration {i}, Loss: {loss(X, Y, w)}')
  return w

"""
The test method will be used to test the model.
"""
def test(X, Y, w):
  total_examples = X.shape[0]
  correct_results = np.sum(classify(X, w) == Y)
  success_percent = correct_results * 100 / total_examples
  print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

print('')
print("STARTING TRAINING")
w = train(data.X_train, data.Y_train, 1e-5, 100)
test(data.X_test, data.Y_test, w)
