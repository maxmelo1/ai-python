from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np

digits  = datasets.load_digits()

decision_tree = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=3)


kf = model_selection.StratifiedKFold(n_splits=10)

predicted_classes = dict()
predicted_classes['tree'] = np.zeros(digits.target.shape[0])
predicted_classes['knn'] = np.zeros(digits.target.shape[0])



inner_kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
outer_kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=None)

param_dist = {'n_neighbors': list(np.arange(1,15)), 'metric': ['euclidean'], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(knn, param_grid=param_dist, cv=outer_kf, scoring='accuracy', refit=False)
grid_search.fit(digits.data, digits.target)
knn_best_params = grid_search.best_params_
print(f'KNN params: {knn_best_params}')

param_dist = {'max_depth': [3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(decision_tree, param_grid=param_dist, cv=outer_kf, scoring='accuracy', refit=False)
grid_search.fit(digits.data, digits.target)
decision_tree_best_params = grid_search.best_params_
print(f'Decision tree params: {decision_tree_best_params}')

knn = KNeighborsClassifier(**knn_best_params)
decision_tree = tree.DecisionTreeClassifier(**decision_tree_best_params)


for train, test in inner_kf.split(digits.data, digits.target):
    data_train, target_train = digits.data[train], digits.target[train]
    data_test, target_test   = digits.data[test], digits.target[test]


    decision_tree.fit(data_train, target_train)
    knn.fit(data_train, target_train)

    decision_tree_predicted = decision_tree.predict(data_test)
    knn_predicted           = knn.predict(data_test)

    predicted_classes['tree'][test] = decision_tree_predicted
    predicted_classes['knn'][test] = knn_predicted

for classifier in predicted_classes.keys():
    print(f'Classifier:  {classifier}. Results\n{metrics.classification_report(digits.target, predicted_classes[classifier])}\n')
    print(f'Confusion matrix: \n{metrics.confusion_matrix(digits.target, predicted_classes[classifier])}\n\n')