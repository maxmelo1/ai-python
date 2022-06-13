from unicodedata import digit
from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection
from sklearn import metrics

import numpy as np

digits  = datasets.load_digits()

decision_tree = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=3)
naive_bayes = GaussianNB()


kf = model_selection.StratifiedKFold(n_splits=10)

predicted_classes = dict()
predicted_classes['tree'] = np.zeros(digits.target.shape[0])
predicted_classes['knn'] = np.zeros(digits.target.shape[0])
predicted_classes['naive'] = np.zeros(digits.target.shape[0])


for train, test in kf.split(digits.data, digits.target):
    data_train, target_train = digits.data[train], digits.target[train]
    data_test, target_test   = digits.data[test], digits.target[test]


    decision_tree.fit(data_train, target_train)
    knn.fit(data_train, target_train)
    naive_bayes.fit(data_train, target_train)

    decision_tree_predicted = decision_tree.predict(data_test)
    knn_predicted           = knn.predict(data_test)
    naive_bayes_predicted   = naive_bayes.predict(data_test)

    predicted_classes['tree'][test] = decision_tree_predicted
    predicted_classes['knn'][test] = knn_predicted
    predicted_classes['naive'][test] = naive_bayes_predicted

for classifier in predicted_classes.keys():
    print(f'Classifier:  {classifier}. Results\n{metrics.classification_report(digits.target, predicted_classes[classifier])}\n')
    print(f'Confusion matrix: \n{metrics.confusion_matrix(digits.target, predicted_classes[classifier])}\n\n')