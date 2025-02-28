import numpy as np
import gzip
import struct

def load_images(filename):
  with gzip.open(filename, 'rb') as f:
    _ignored, n_images, rows, cols = struct.unpack('>IIII', f.read(16))
    all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return all_pixels.reshape(n_images, rows * cols)

def prepend_bias(X):
  return np.insert(X, 0, 1, axis=1)

X_train = prepend_bias(load_images('mnist/train-images-idx3-ubyte.gz'))

X_test = prepend_bias(load_images('mnist/t10k-images-idx3-ubyte.gz'))

print(X_train.shape, X_test.shape)