"""
    Premier exercice de notre quatrième cours de machine learning.
    Le but était d'estimer e avec a^(x) = a^(x) * ln(a) pour ln(a) allant vers 1
"""

import numpy as np

def main():
    """
        ...
    """
        
    a = 1
    step = 1
    i = 1

    while(round(np.log(a), 100) != 1.0):
        if(np.log(a) > 1):
            if(np.log(a - step) < 1):
                step = step/i
            a -= step
        else:
            if(np.log(a + step) > 1):
                step = step/i
            a += step
        i += 1
        print(np.log(a))
    print(a)

if __name__ == "__main__":
    main()
