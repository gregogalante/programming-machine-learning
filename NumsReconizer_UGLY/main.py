import mnist as data
from classifier import Classifier

classifiers = []

for i in range(10):
  print(f'Training classifier {i}...')
  Y_train = data.Y_trains[i]
  classifier = Classifier(data.X_train, Y_train)
  classifier.train()
  classifiers.append(classifier)

for i in range(10):
  print(f'Testing classifier {i}...')
  Y_test = data.Y_tests[i]
  classifiers[i].test(data.X_test, Y_test)

def classify_image(x):
  value = None
  for i in range(10):
    print(f'Executing classifier {i}...')
    if classifiers[i].execute(x) == 1:
      print(f'Image is a {i}')
      value = i
      break
  return value

# Test the classifier on a single image
random_image = data.X_test[0]
classify_image(random_image)
