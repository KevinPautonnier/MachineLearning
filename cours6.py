"""
    Exercice de notre sixième cours de machine learning.
    Le but était d'utiliser d'utiliser le random forest.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    """
        ...
    """

    digits = datasets.load_digits()

    data = digits['data']
    target = digits['target']

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    clf.fit(x_train, y_train)

    correct_predictions = 0
    number_of_values_tested = 0
    for key, x in enumerate(x_test):
        if clf.predict([x]) == y_test[key]:
            correct_predictions += 1
        number_of_values_tested += 1

    print("precision : ", (correct_predictions/number_of_values_tested)*100)

if __name__ == "__main__":
    main()
