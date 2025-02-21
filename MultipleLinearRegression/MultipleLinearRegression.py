import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
The MultipleLinearRegression class will be used to build and use a model to predict data Y based on data X.
"""
class MultipleLinearRegression:
  """
  The __init__ method will be used to initialize the MultipleLinearRegression object.

  The x_csv_path parameter is a string that represents the path to the CSV file that contains the X data.
  The y_csv_path parameter is a string that represents the path to the CSV file that contains the Y data.

  NOTE: Bot CSV files should have the same number of rows.
  """
  def __init__(self, x_csv_path, y_csv_path):
    self.X = np.loadtxt(x_csv_path, delimiter=',', skiprows=1)
    self.Y = np.loadtxt(y_csv_path, delimiter=',', skiprows=1)

    # add bias term to X
    self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

    # reshape Y
    self.Y = self.Y.reshape((self.Y.shape[0], 1))

    print('X shape:', self.X.shape)
    print('Y shape:', self.Y.shape)

  """
  The predict method will be used to predict the Y data based on the X data.
  """
  def predict(self, X, w):
    return np.matmul(X, w)

  """
  The loss method will be used to calculate the loss of the model.
  """
  def loss(self, w):
    return np.average((self.predict(self.X, w) - self.Y) ** 2)

  """
  The gradient method will be used to calculate the gradient of the model.
  """
  def gradient(self, w):
    return 2 * np.matmul(self.X.T, (self.predict(self.X, w) - self.Y)) / self.X.shape[0]

  """
  The train method will be used to train the model.
  """
  def train(self, learning_rate=0.001, epochs=100000):
    w = np.zeros((self.X.shape[1], 1))
    for i in range(epochs):
      w -= learning_rate * self.gradient(w)
      if i % 100 == 0:
        print('Epoch:', i, 'Loss:', self.loss(w))
    return w

  """
  The plot method will be used to plot the predictions and real values.
  """
  def plot(self, w):
    plt.figure()
    plt.plot(self.predict(self.X, w), label='Predictions')
    plt.plot(self.Y, label='Real values')
    plt.legend()
    plt.show()


