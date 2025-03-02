import mnist as data
from classifier import Classifier

classifier = Classifier(data.X_train, data.Y_train)
classifier.train(1e-5, 100)

classifier.test(data.X_test, data.Y_test)