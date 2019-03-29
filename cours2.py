import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

points = [(1,7), (2,3), (3,1)]
teta0 = next_0 =1
teta1 = next_1 =1

step = 1

min_cost = 100

df_t_0 = lambda t_0,T_1: reduce(lambda x,y : x+y, [(t_0 + T_1 * elem[0]) - elem[1] for elem in points])
df_t_1 = lambda t_0,T_1: reduce(lambda x,y : x+y, [((t_0 + T_1 * elem[0]) - elem[1]) * elem[0] for elem in points])
df_j_0 = lambda t_0,T_1: reduce(lambda x,y : x+y, [((t_0 + T_1 * elem[0]) - elem[1])**2 for elem in points])

while step < 2000000 :
    learning_rate = 1/step
    teta0 = next_0
    teta1 = next_1
    next_0 = teta0 - learning_rate/len(points) * df_t_0(teta0,teta1)
    next_1 = teta1 - learning_rate/len(points) * df_t_1(teta0,teta1)

    cost = 1/(2*len(points)) * df_j_0(teta0,teta1)
    
    if(cost < 0.8):
        break
    if(cost < min_cost):
        min_cost = cost

    step+=1

plt.subplot(211)
x,y = [elem[0] for elem in points], [elem[1] for elem in points]
plt.scatter(x,y)

lol = np.arange(0,3,0.1)
y_chapeau = [next_0 + next_1 * elem for elem in lol]
plt.plot(lol, y_chapeau)

plt.subplot(212)
y_lol = [0] * 30

x,y =  [elem[0] for elem in points], [ elem[1] - (next_0 + next_1 * elem[0]) for elem in points]
plt.scatter(x,y)
plt.plot(lol, y_lol)
plt.show()

print(step)
print(min_cost)
