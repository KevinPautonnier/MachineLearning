"""
    Exercice de notre huitième cours de machine learning.
    Le but était d'utiliser d'utiliser le regression logistic, random forest, SVM, One vs One et One vs All.
    precision recal et time
"""

import timeit
import functools
from sklearn import svm
from cours3 import o_v_o, o_v_r
from cours6 import random_forest
from itertools import combinations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import operator

FACES = datasets.fetch_olivetti_faces()
DATA = FACES['data']
TARGET = FACES['target']

#create list of all combination possible with digit from 0 to 9
COMB = list(combinations(set(TARGET), 2))

def svm_train(x_train, y_train):
    """
        train the random forest
    """

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(x_train, y_train)

    return clf

def svm_test(clf, x_test, y_test):
    """
        create test data for random forest
    """

    correct_predictions = 0
    number_of_values_tested = 0
    for key, x in enumerate(x_test):
        if clf.predict([x]) == y_test[key]:
            correct_predictions += 1
        number_of_values_tested += 1

    return correct_predictions, number_of_values_tested

def s_v_m(x_train, x_test, y_train, y_test):
    """
        random forest function
    """

    time_to_train = 0
    clf = svm_train(x_train, y_train)
    time_to_train += timeit.timeit(functools.partial(svm_train, x_train, y_train), number=1)

    time_to_test = 0
    correct_predictions, number_of_values_tested = svm_test(clf, x_test, y_test)
    time_to_test += timeit.timeit(functools.partial(svm_test, clf, x_test, y_test), number=1)

    percent_good_prediction = (correct_predictions/number_of_values_tested)*100

    return percent_good_prediction, time_to_train, time_to_test

def main():
    """
        ...
    """

    x_train, x_test, y_train, y_test = train_test_split(DATA,
                                                        TARGET,
                                                        test_size=0.33,
                                                        random_state=42)

    prct_predict = o_v_o(x_train, x_test, y_train, y_test, COMB, (set(TARGET)))
    print("One versus one")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = o_v_r(x_train, x_test, y_train, y_test, (set(TARGET)))
    print("One versus rest")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = random_forest(x_train, x_test, y_train, y_test)
    print("Random forest")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = s_v_m(x_train, x_test, y_train, y_test)
    print("SVM")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

if __name__ == "__main__":
    main()
