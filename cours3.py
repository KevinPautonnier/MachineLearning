
"""
    Troisième exercice de notre cours de machine learning.
    Le but était d'utiliser des classifiers en One vs One et One vs Rest
    sur une base de chiffre manuscrit pour apprendre à les deviner
"""

import timeit
import functools
from itertools import combinations
import operator
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

def ovo_classifier(elem, x_train, y_train):
    """
        create one vs one classifier
    """
    img0 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem[0]]
    img1 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem[1]]

    y = [1 for elem in img0]
    y += [0 for elem in img1]

    return LogisticRegression(solver='lbfgs', max_iter=1000).fit(img0 + img1, y)

def ovo_test(dic, x_test, y_test, nb_classes):
    """
        test one vs one classifier
    """
    correct_predict = 0
    for index, value in enumerate(x_test):
        predict = {elem : 0 for elem in nb_classes}
        for key in dic:
            predict[key[0] if dic[key].predict([value]) else key[1]] += 1
        predicted_value = max(predict.items(), key=operator.itemgetter(1))[0]
        if predicted_value == y_test[index]:
            correct_predict += 1
    
    return correct_predict

def o_v_o(x_train, x_test, y_train, y_test, comb, nb_classes):
    """
        One vs one logic
    """
    dic = {}
    time_to_train = 0
    #fill my dictionnary with each combination and his calculate classifier for the OvO
    for elem in comb:
        dic[elem] = ovo_classifier(elem, x_train, y_train)
        time_to_train += timeit.timeit(functools.partial(ovo_classifier, elem, x_train, y_train), number=1)

    time_to_test = 0
    correct_predict = ovo_test(dic, x_test, y_test, nb_classes)
    time_to_test += timeit.timeit(functools.partial(ovo_test, dic, x_test, y_test, nb_classes), number=1)

    #calculate the ratio of good prediction
    prct_predict = (correct_predict/len(x_test))*100

    return prct_predict, time_to_train, time_to_test

def ovr_classifier(elem, x_train, y_train):
    """
        create one vs rest classifier
    """
    img0 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem]
    img1 = [x_train[idx] for idx, e in enumerate(y_train) if e != elem]

    y = [1 for elem in img0]
    y += [0 for elem in img1]

    return LogisticRegression(solver='lbfgs', max_iter=1000).fit(img0 + img1, y)
    
def ovr_test(dic, x_test, y_test, nb_classes):
    """
        test one vs rest classifier
    """

    correct_predict = 0
    for index, value in enumerate(x_test):
        predict = {elem : 0 for elem in nb_classes}
        for key in dic:
            predict[key] = dic[key].predict_proba([value])[0][1]

        predicted_value = max(predict.items(), key=operator.itemgetter(1))[0]
        if predicted_value == y_test[index]:
            correct_predict += 1
    
    return correct_predict

def o_v_r(x_train, x_test, y_train, y_test, nb_classes):
    """
        One vs Rest logic
    """
    dic = {}
    time_to_train = 0
    #fill my dictionnary with each combination and his calculate classifier for the OvR
    for elem in nb_classes:
        dic[elem] = ovr_classifier(elem, x_train, y_train)
        time_to_train += timeit.timeit(functools.partial(ovr_classifier, elem, x_train, y_train), number=1)

    time_to_test = 0
    correct_predict = ovr_test(dic, x_test, y_test, nb_classes)
    time_to_test += timeit.timeit(functools.partial(ovr_test, dic, x_test, y_test, nb_classes), number=1)

    #calculate the ratio of good prediction
    prct_predict = (correct_predict/len(x_test))*100

    return prct_predict, time_to_train, time_to_test

def main():
    """
        ...
    """
    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.33, random_state=42)

    
    #create list of all combination possible with digit from 0 to 9
    comb = list(combinations(set(TARGET), 2))
    
    prct_predict = o_v_o(x_train, x_test, y_train, y_test, comb, set(TARGET))
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")

    prct_predict = o_v_r(x_train, x_test, y_train, y_test, set(TARGET))
    print(f"Predict ratio : {prct_predict[0]:.2f}% Time to train : {prct_predict[1]:.2f} Time to test : {prct_predict[2]:.2f}")
    
if __name__ == "__main__":
    main()
