import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def forward(X, w):
  weighted_sum = np.matmul(X, w)
  return sigmoid(weighted_sum)

def classify(X, w):
  return np.round(forward(X, w))

def mse_loss(X, Y, w):
  return np.average((forward(X, w) - Y)**2)

def loss(X, Y, w):
  y_hat = forward(X, w)
  first_term = Y * np.log(y_hat)
  second_term = (1 - Y) * np.log(1 - y_hat)
  return -np.average(first_term + second_term)

def gradient(X, Y, w):
  return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
  w = np.zeros((X.shape[1], 1))
  for i in range(iterations):
    w -= lr * gradient(X, Y, w)
    if i % 1000 == 0:
      print(f'Iteration {i}, Loss: {loss(X, Y, w)}')
  return w

def test(X, Y, w):
  total_examples = X.shape[0]
  correct_results = np.sum(classify(X, w) == Y)
  success = correct_results * 100 / total_examples
  print(f'Total Examples: {total_examples}')
  print(f'Correct Results: {correct_results}')

x1, x2, x3, y = np.loadtxt('police.txt', skiprows=1, unpack=True)
X = np.column_stack((x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, 10000, 0.1)

test(X, Y, w)