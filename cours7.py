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

    prct_predict = o_v_o(x_train, x_test, y_train, y_test, COMB, set(TARGET))
    print("One versus one")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = o_v_r(x_train, x_test, y_train, y_test, set(TARGET))
    print("One versus rest")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = random_forest(x_train, x_test, y_train, y_test)
    print("Random forest")
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

if __name__ == "__main__":
    main()