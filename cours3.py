import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()

data = digits['data']
target  = digits['target']

def OvO(x):
    img0 = [data[idx] for idx,e in enumerate(target) if e == x]
    for index in range(0, 10):
        if(index != x):
            img1 = [data[idx] for idx,e in enumerate(target) if e == index]
                        
            index_learn_0 = int(len(img0)*0.8)
            index_test_0 = index_learn_0+1

            index_learn_1 = int(len(img1)*0.8)
            index_test_1 = index_learn_1+1

            learn0, test0 = img0[index_learn_0:], img0[:index_test_0]
            learn1, test1 = img1[index_learn_1:], img1[:index_test_1]

            value0 = [0] * len(learn0)
            value1 = [1] * len(learn1)

            learn01 = learn0 + learn1
            value01 = value0 + value1

            reg = LogisticRegression(solver='lbfgs').fit(learn01, value01)

            #print(reg.score(learn01, value01))
            print("Predicted :" , x if reg.predict([test1[0]]) else " other value", "Real value : ", x)


def OvR(x):
    img0 = [data[idx] for idx,e in enumerate(target) if e == x]
    img1 = [data[idx] for idx,e in enumerate(target) if e != x]
                
    index_learn_0 = int(len(img0)*0.8)
    index_test_0 = index_learn_0+1

    index_learn_1 = int(len(img1)*0.8)
    index_test_1 = index_learn_1+1

    learn0, test0 = img0[index_learn_0:], img0[:index_test_0]
    learn1, test1 = img1[index_learn_1:], img1[:index_test_1]

    value0 = [0] * len(learn0)
    value1 = [1] * len(learn1)

    learn01 = learn0 + learn1
    value01 = value0 + value1

    reg = LogisticRegression(solver='lbfgs').fit(learn01, value01)

    #print(reg.score(learn01, value01))
    print("Predicted :" , x if reg.predict([test1[0]]) else " other value", "Real value : ", x)

for i in range(0, 10):
    OvR(i)
    OvR(i)
