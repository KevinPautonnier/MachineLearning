"""
    Exercice de notre septième cours cours de machine learning.
    Le but était d'utiliser d'utiliser le random forest, One vs One et One vs All
    pour comparer leur performance
"""

from cours3 import o_v_o, o_v_r
from cours6 import random_forest
from sklearn import datasets
from itertools import combinations
from sklearn.model_selection import train_test_split

DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

COMB = list(combinations(set(TARGET), 2))

def main():
    """
        ...
    """

    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.33, random_state=42)

    prct_predict, time_to_train, time_to_test, prec, rec, F1 = o_v_o(x_train, x_test, y_train, y_test, COMB, set(TARGET)) 
    print("One versus one")
    print(f"Predict ratio : {prct_predict:.2f}% Time to train : {time_to_train:.2f} Time to test : {time_to_test:.2f}")
    print(f"Precision : {prec:.2f}% Recall : {rec:.2f} F1_score : {F1:.2f}")

    prct_predict, time_to_train, time_to_test, prec, rec, F1 = o_v_r(x_train, x_test, y_train, y_test, set(TARGET))
    print("One versus rest")
    print(f"Predict ratio : {prct_predict:.2f}% Time to train : {time_to_train:.2f} Time to test : {time_to_test:.2f}")
    print(f"Precision : {prec:.2f}% Recall : {rec:.2f} F1_score : {F1:.2f}")

    prct_predict, time_to_train, time_to_test, prec, rec, F1 = random_forest(x_train, x_test, y_train, y_test)
    print("Random forest")
    print(f"Predict ratio : {prct_predict:.2f}% Time to train : {time_to_train:.2f} Time to test : {time_to_test:.2f}")
    print(f"Precision : {prec:.2f}% Recall : {rec:.2f} F1_score : {F1:.2f}")

if __name__ == "__main__":
    main()