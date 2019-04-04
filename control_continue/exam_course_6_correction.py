import numpy as np
from sklearn.linear_model import LinearRegression
import time
import datetime
import matplotlib.pyplot as plt

date = ["2019-03-16", "2019-03-18", "2019-03-20", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-30", "2019-04-03"]
values = [81682, 81720, 81760, 81826, 81844, 81864, 81881, 81900, 81933, 82003]

#transform date in timestemp
timestamp_date = [time.mktime(datetime.datetime.strptime(elem, "%Y-%m-%d").timetuple()) for elem in date]

points = []

X = np.array(timestamp_date).reshape(-1, 1)
y = np.array(values).reshape(-1, 1)

reg = LinearRegression().fit(X, y)

predict = np.array(time.mktime(datetime.datetime.strptime("2019-04-04", "%Y-%m-%d").timetuple())).reshape(-1, 1)

result = reg.predict(predict)
result2 = [reg.predict(np.array(elem).reshape(-1, 1))[0][0] for elem in timestamp_date]

plt.scatter(X,y)
plt.plot(predict, result, 'ro')
plt.plot(timestamp_date, result2)
plt.show()