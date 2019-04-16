"""
    Deuxième exercice de notre quatrième cours de machine learning.
    Le but était d'estiùmer e grâce à la séquence de taylor.
"""

import numpy as np
import matplotlib.pyplot as plt

def fact(nb):
    """
        factoriel function
    """
    if(nb >= 1):
        return nb * fact(nb-1)
    else: 
        return 1

def main():
    """
        ...
    """
        
    k_max = 12
    total = 0
    x = 1
    plots = {}

    #taylor sequence
    for k in range(0, k_max, 1):
        total += x ** k / fact(k)
        plots[k] = total
        print(k)
        print(total)

    #the graph show an estimation of e
    x = plots.keys()
    y = plots.values()
    plt.plot(x, y)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
