import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
The Classification class will be used to build and use a model to classify data Y based on data X.
"""
class Classification:
  def __init__(self, x_csv_path, y_csv_path):
    self.X = np.loadtxt(x_csv_path, delimiter=',', skiprows=1)
    self.Y = np.loadtxt(y_csv_path, delimiter=',', skiprows=1)

  # TODO: Continue from here...