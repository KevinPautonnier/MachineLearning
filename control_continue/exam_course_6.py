import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

dates = ["2019-03-16", "2019-03-18", "2019-03-20", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-30", "2019-04-03"]
values = [81682, 81720, 81760, 81826, 81844, 81864, 81881, 81900, 81933, 82003]

points = []

for index, date in dates:
    points.append((date, values[index]))

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

#need to transform date in timestemp