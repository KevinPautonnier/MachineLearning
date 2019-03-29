import numpy as np

a = 1
step = 1
i = 1

while(round(np.log(a), 100) != 1.0):
    if(np.log(a)>1):
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