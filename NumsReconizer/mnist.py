import numpy as np
import gzip
import struct

def load_labels(filename):
  with gzip.open(filename, 'rb') as f:
    # skip the header bytes
    f.read(8)
    # read all the labels
    all_labels = f.read()
    # re-shape the labels into a 1 column matrix
    return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

def load_images(filename):
  with gzip.open(filename, 'rb') as f:
    # read the header information
    _ignored, n_images, rows, cols = struct.unpack('>IIII', f.read(16))
    # read all the pixels into a single numpy array
    all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
    # re-shape the pixels into a matrix where each row is an image
    return all_pixels.reshape(n_images, rows * cols)

def prepend_bias(X):
  return np.insert(X, 0, 1, axis=1)

def one_hot_encode(Y):
  n_labels = Y.shape[0]
  n_classes = 10
  encoded_Y = np.zeros((n_labels, n_classes))
  for i in range(n_labels):
    label = Y[i, 0]
    encoded_Y[i, label] = 1
  return encoded_Y

print('Loading MNIST X_train...')
X_train = prepend_bias(load_images('mnist/train-images-idx3-ubyte.gz'))
print(X_train.shape)

print('Loading MNIST X_test...')
X_test = prepend_bias(load_images('mnist/t10k-images-idx3-ubyte.gz'))
print(X_test.shape)

print('Loading MNIST Y_train...')
Y_train = one_hot_encode(load_labels('mnist/train-labels-idx1-ubyte.gz'))
print(Y_train.shape)

print('Loading MNIST Y_test...')
Y_test = load_labels('mnist/t10k-labels-idx1-ubyte.gz')
print(Y_test.shape)

