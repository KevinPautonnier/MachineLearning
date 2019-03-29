import numpy as np
import matplotlib.pyplot as plt

k_max = 12
total = 0
x = 1
plots = {}

def fact(nb):
    if(nb >= 1):
        return nb * fact(nb-1)
    else: 
        return 1

for k in range(0,k_max,1):
    total += x ** k / fact(k)
    plots[k] = total
    print(k)
    print(total)

x = plots.keys()
y = plots.values()
plt.plot(x,y)

plt.tight_layout()
plt.show()