"""
    Second exercice de notre cours de machine learning.
    Le but était de coder une fonction de régression linéaire
    et de l'utiliser sur une fonction de coup pour se rapprocher
    de notre hypothese
"""

from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

#init data
POINTS = [(1, 7), (2, 3), (3, 1)]

def calculate_teta_0(t_0, t_1):
    """
        teta zero function
    """
    return reduce(lambda x, y: x+y, [(t_0 + t_1 * elem[0]) - elem[1] for elem in POINTS])

def calculate_teta_1(t_0, t_1):
    """
        teta one function
    """
    return reduce(lambda x, y: x+y, [((t_0 + t_1 * elem[0]) - elem[1]) * elem[0] for elem in POINTS])

def calculate_job(t_0, t_1):
    """
        job function
    """
    return reduce(lambda x, y: x+y, [((t_0 + t_1 * elem[0]) - elem[1])**2 for elem in POINTS])

def main():
    """
        Calculate the linear regression for the points and show it on graph
    """
    
    teta_0 = next_0 = 1
    teta_1 = next_1 = 1
    step = 1
    min_cost = 100

    while step < 2000000:
        learning_rate = 1/step
        teta_0 = next_0
        teta_1 = next_1
        next_0 = teta_0 - learning_rate/len(POINTS) * calculate_teta_0(teta_0, teta_1)
        next_1 = teta_1 - learning_rate/len(POINTS) * calculate_teta_1(teta_0, teta_1)

        #calculate the cost for teta0 and teta1 with the job function
        cost = 1/(2*len(POINTS)) * calculate_job(teta_0, teta_1)
        
        if(cost < 0.8):
            break
        if(cost < min_cost):
            min_cost = cost

        step += 1

    plt.subplot(211)
    x, y = [elem[0] for elem in POINTS], [elem[1] for elem in POINTS]
    plt.scatter(x, y)

    lol = np.arange(0, 3, 0.1)
    y_chapeau = [next_0 + next_1 * elem for elem in lol]
    plt.plot(lol, y_chapeau)

    plt.subplot(212)
    y_lol = [0] * 30

    x, y = [elem[0] for elem in POINTS], [elem[1] - (next_0 + next_1 * elem[0]) for elem in POINTS]
    plt.scatter(x, y)
    plt.plot(lol, y_lol)
    plt.show()

    print(step)
    print(min_cost)

if __name__ == "__main__":
    main()
