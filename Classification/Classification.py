import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
The Classification class will be used to build and use a model to classify data Y based on data X.
"""
class Classification:
  """
  The __init__ method will be used to initialize the Classification object.

  The x_csv_path parameter is a string that represents the path to the CSV file that contains the X data.
  The y_csv_path parameter is a string that represents the path to the CSV file that contains the Y data.

  NOTE: Bot CSV files should have the same number of rows.
  """
  def __init__(self, x_csv_path, y_csv_path):
    self.X = np.loadtxt(x_csv_path, delimiter=',', skiprows=1)
    self.Y = np.loadtxt(y_csv_path, delimiter=',', skiprows=1)

    # reshape Y
    self.Y = self.Y.reshape((self.Y.shape[0], 1))

    print('X shape:', self.X.shape)
    print('Y shape:', self.Y.shape)

  """
  The sigmoid method will be used to calculate the sigmoid of a given value.
  """
  def sigmoid(self, z):
    return 1/(1 + np.exp(-z))

  """
  The forward method will be used to calculate the forward pass of the model.
  """
  def forward(self, X, w):
    weighted_sum = np.matmul(X, w)
    return self.sigmoid(weighted_sum)

  """
  The classify method will be used to classify the data based on the forward pass.
  """
  def classify(self, X, w):
    return np.round(self.forward(X, w))

  """
  The loss method will be used to calculate the loss of the model.
  """
  def loss(self, X, Y, w):
    y_hat = self.forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

  """
  The gradient method will be used to calculate the gradient of the model.
  """
  def gradient(self, X, Y, w):
    return np.matmul(X.T, (self.forward(X, w) - Y)) / X.shape[0]

  """
  The train method will be used to train the model.
  """
  def train(self, learning_rate=0.001, epochs=100000):
    w = np.zeros((self.X.shape[1], 1))
    for i in range(epochs):
      w -= learning_rate * self.gradient(self.X, self.Y, w)
      if i % 1000 == 0:
        print(f'Iteration {i}, Loss: {self.loss(self.X, self.Y, w)}')
    return w

  """
  The plot method will be used to plot the predictions and real values.
  """
  def plot(self, w):
    plt.figure()
    plt.plot(self.classify(self.X, w), label='Classifications')
    plt.plot(self.Y, label='Real values')
    plt.legend()
    plt.show()