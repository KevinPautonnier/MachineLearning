
"""
    Troisième exercice de notre cours de machine learning.
    Le but était d'utiliser des classifiers en One vs One et One vs Rest
    sur une base de chiffre manuscrit pour apprendre à les deviner
"""

from itertools import combinations
import operator
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def OvO(elem, dic, x_train, y_train):
    """
        create one vs one classifier
    """
    img0 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem[0]]
    img1 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem[1]]

    y = [1 for elem in img0]
    y += [0 for elem in img1]

    dic[elem] = LogisticRegression(solver='lbfgs').fit(img0+img1,y)

def OvR(elem, dic, x_train, y_train):
    """
        create one vs rest classifier
    """
    img0 = [x_train[idx] for idx, e in enumerate(y_train) if e == elem]
    img1 = [x_train[idx] for idx, e in enumerate(y_train) if e != elem]

    y = [1 for elem in img0]
    y += [0 for elem in img1]

    dic[elem] = LogisticRegression(solver='lbfgs').fit(img0+img1,y)

def main():
    """
        ...
    """
        
    digits = datasets.load_digits()

    data = digits['data']
    target = digits['target']

    #create list of all combination possible with digit from 0 to 9
    comb = list(combinations(set(target), 2))

    dic = {}
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    #fill my dictionnary with each combination and his calculate classifier for the OvO
    for elem in comb:
        OvO(elem, dic, X_train, y_train)

    correct_predict = 0
    for index, value in enumerate(X_test):
        predict = {elem : 0 for elem in range(0,10)}
        for key in dic:
            predict[key[0] if dic[key].predict([value]) else key[1]] += 1
        predicted_value = max(predict.items(), key=operator.itemgetter(1))[0]
        if(predicted_value == y_test[index]): correct_predict += 1
    
    #calculate the ratio of good prediction
    prct_predict = (correct_predict/len(X_test))*100
    print(f"Predict ratio : {prct_predict:.2f}")

    dic = {}
    #fill my dictionnary with each combination and his calculate classifier for the OvR
    for elem in set(target):
        OvR(elem, dic, X_train, y_train)

    correct_predict = 0
    for index, value in enumerate(X_test):
        predict = {elem : 0 for elem in range(0, 10)}
        for key in dic:
            predict[key] = dic[key].predict_proba([value])[0][1]

        predicted_value = max(predict.items(), key=operator.itemgetter(1))[0]
        if(predicted_value == y_test[index]): correct_predict += 1  
    
    #calculate the ratio of good prediction
    prct_predict = (correct_predict/len(X_test))*100
    print(f"Predict ratio : {prct_predict:.2f}")

if __name__ == "__main__":
    main()
