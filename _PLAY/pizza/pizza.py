import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# VERSION 1: Supervised Learning with Linear Regression.
####################################################################################################
# M = "y = x * w"
# M = model
# w = weight

# def predict(X, w):
#   return X * w

# # The loss function is used to measure the distance between the current model and the "ground truth".
# # Lower values are better.
# def loss(X, Y, w):
#   return np.average((predict(X, w) - Y) ** 2)

# def train(X, Y, iterations, lr):
#   w = 0
#   for i in range(iterations):
#     current_loss = loss(X, Y, w)
#     print("Iteration %4d => Loss: %.6f" % (i, current_loss))
#     if loss(X, Y, w + lr) < current_loss:
#       w += lr
#     elif loss(X, Y, w - lr) < current_loss:
#       w -= lr
#     else:
#       return w
#   raise Exception("Couldn't converge within %d iterations" % iterations)

# X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)
# w = train(X, Y, 10000, 0.01)

# sns.set()
# plt.axis([0, 50, 0, 50])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Reservations', fontsize=30)
# plt.ylabel('Pizzas', fontsize=30)
# plt.plot(X, Y, 'bo')
# plt.plot([0, 50], [0, 50 * w], 'g', linewidth=2)
# plt.show()

# VERSION 2: Supervised Learning with Linear Regression using Bias.
####################################################################################################
# M = "y = x * w + b"
# M = model
# w = weight
# b = bias

# def predict(X, w, b):
#   return X * w + b

# # The loss function is used to measure the distance between the current model and the "ground truth".
# # Lower values are better.
# def loss(X, Y, w, b):
#   return np.average((predict(X, w, b) - Y) ** 2)

# def train(X, Y, iterations, lr):
#   w = b = 0
#   for i in range(iterations):
#     current_loss = loss(X, Y, w, b)
#     print("Iteration %4d => Loss: %.6f" % (i, current_loss))
#     if loss(X, Y, w + lr, b) < current_loss:
#       w += lr
#     elif loss(X, Y, w - lr, b) < current_loss:
#       w -= lr
#     elif loss(X, Y, w, b + lr) < current_loss:
#       b += lr
#     elif loss(X, Y, w, b - lr) < current_loss:
#       b -= lr
#     else:
#       return w, b
#   raise Exception("Couldn't converge within %d iterations" % iterations)

# X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)
# w, b = train(X, Y, 10000, 0.01)

# sns.set()
# plt.axis([0, 50, 0, 50])
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Reservations', fontsize=30)
# plt.ylabel('Pizzas', fontsize=30)
# plt.plot(X, Y, 'bo')
# plt.plot([0, 50], [b, 50 * w + b], 'g', linewidth=2)
# plt.show()

# VERSION 3: Gradient Descent (one dimension).
####################################################################################################
# Gradient Descent is a technique used to find the minimum of the loss function in a faster and more precise way.
# PROBLEM: Our train function needs to manage a number of conditions based on the number of parameters.
# The relationship between the number of parameters and the number of conditions is exponential (2^n).
# If we use this code for real use cases that require a large number of parameters, the code will be very slow.

# def predict(X, w, b):
#   return X * w + b

# def gradient(X, Y, w):
#   return 2 * np.average(X * (predict(X, w, 0) - Y))

# def train(X, Y, iterations, lr):
#   w = 0
#   for i in range(iterations):
#     w -= gradient(X, Y, w) * lr
#   return w

# X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)
# w = train(X, Y, 100, 0.001)
# print(w)

# VERSION 4: Gradient Descent (two dimensions).
####################################################################################################

def predict(X, w, b):
  return X * w + b

def gradient(X, Y, w, b):
  w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
  b_gradient = 2 * np.average(predict(X, w, b) - Y)
  return w_gradient, b_gradient

def train(X, Y, iterations, lr):
  w = b = 0
  for i in range(iterations):
    w_gradient, b_gradient = gradient(X, Y, w, b)
    w -= w_gradient * lr
    b -= b_gradient * lr
  return w, b

X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)
w, b = train(X, Y, 100, 0.001)
print(w, b)
