from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV



def split(embeddings, nodes):
    return train_test_split(embeddings, nodes, test_size=0.4, random_state=42)


def train(X, y):
    # HYPERPARAMETERS = {'C': [0.1,0.5,1,1.5,2]}
    # clf = GridSearchCV(svm.LinearSVC(), HYPERPARAMETERS, refit=True, verbose=2)

    # X, y = make_classification(n_samples=1000, n_features=4,
    # n_informative = 2, n_redundant = 0,
    # random_state = 0, shuffle = False)

    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf.fit(X, y)

    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # clf.fit(X, y)

    # metrics = ['euclidean', 'manhattan']
    # neighbors = np.arange(1, 16)
    # param_grid = dict(metric=metrics, n_neighbors=neighbors)
    # cross_validation_fold = 10
    # param_grid

    # grid_search = GridSearchCV(clf, param_grid, cv=cross_validation_fold, scoring='accuracy', refit=True)
    # grid_search.fit(X, y)

    # clf = KNeighborsClassifier(n_neighbors=5)
    # clf = KNeighborsClassifier(n_neighbors=2, algorithm = 'ball_tree', leaf_size=50, weights='uniform', p=1)
    # clf.fit(X, y)
    clf = svm.LinearSVC()
    clf.fit(X, y)

    # return  grid_search
    return clf
    # return neigh


def test(X, y, clf):
    # scores = cross_val_score(clf, X, y, cv=5)
    predictions = clf.predict(X)
    return metrics.accuracy_score(y, predictions)
    # return accuracy_score(y, predictions)
    # return scores.mean()
