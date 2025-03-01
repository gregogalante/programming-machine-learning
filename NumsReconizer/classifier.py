import numpy as np

class Classifier:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.w = None

  def _sigmoid(self, z):
    return 1/(1 + np.exp(-z))
  
  def _forward(self, X, w):
    weighted_sum = np.matmul(X, w)
    return self._sigmoid(weighted_sum)
  
  def _classify(self, X, w):
    return np.round(self._forward(X, w))
  
  def _loss(self, X, Y, w):
    y_hat = self._forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)
  
  def _gradient(self, X, Y, w):
    return np.matmul(X.T, (self._forward(X, w) - Y)) / X.shape[0]
  
  def train(self, learning_rate=1e-5, epochs=100):
    w = np.zeros((self.X.shape[1], 1))
    for i in range(epochs):
      w -= learning_rate * self._gradient(self.X, self.Y, w)
      if i % 10 == 0:
        print(f'Iteration {i}, Loss: {self._loss(self.X, self.Y, w)}')
    self.w = w
    return w
  
  def test(self, X, Y):
    if self.w is None:
      raise Exception("Model has not been trained yet.")
    total_examples = X.shape[0]
    correct_results = np.sum(self._classify(X, self.w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

  def execute(self, X):
    if self.w is None:
      raise Exception("Model has not been trained yet.")
    return self._classify(X, self.w)