"""
    Exercice de notre sixième cours de machine learning.
    Le but était d'utiliser d'utiliser le random forest.
"""

import timeit
import functools
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from stats_functions import *

DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

def random_forest_train(x_train, y_train):
    """
        create random forest classifier and fit the datas
    """
    clf = RandomForestClassifier(n_estimators=100,
                                 random_state=42)
    clf.fit(x_train, y_train)

    return clf

def random_forest_test(clf, x_test, y_test):
    """
        test random forest classifier and return the correct predictions
    """

    TP = FP = TN = FN = 0
    correct_predictions = 0
    number_of_values_tested = 0
    for key, x in enumerate(x_test):
        if clf.predict([x]) == y_test[key]:
            correct_predictions += 1
            TP += 1
        else:
            FP += 1
        number_of_values_tested += 1

    return correct_predictions, number_of_values_tested, TP, FP, TN, FN

def random_forest(x_train, x_test, y_train, y_test):
    """
        random forest function
    """

    time_to_train = 0
    clf = random_forest_train(x_train, y_train)
    time_to_train += timeit.timeit(functools.partial(random_forest_train, x_train, y_train), number=1)

    time_to_test = 0
    correct_predictions, number_of_values_tested, TP, FP, TN, FN = random_forest_test(clf, x_test, y_test)
    time_to_test += timeit.timeit(functools.partial(random_forest_test, clf, x_test, y_test), number=1)

    percent_prediction = (correct_predictions/number_of_values_tested)*100

    prec = precision(TP, FP)
    rec = recall(TP, FN)
    F1 = F1_score(prec, rec)

    return percent_prediction, time_to_train, time_to_test, prec, rec, F1

def main():
    """
        ...
    """

    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.33, random_state=42)

    prct_predict, time_to_train, time_to_test, prec, rec, F1 = random_forest(x_train, x_test, y_train, y_test)
    print("Random forest")
    print(f"Predict ratio : {prct_predict:.2f}% Time to train : {time_to_train:.2f} Time to test : {time_to_test:.2f}")
    print(f"Precision : {prec:.2f}% Recall : {rec:.2f} F1_score : {F1:.2f}")

if __name__ == "__main__":
    main()
