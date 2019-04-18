from cours3 import o_v_o, o_v_r
from sklearn import datasets
from sklearn.model_selection import train_test_split

DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

x_train, x_test, y_train, y_test = train_test_split(
    DATA, TARGET, test_size=0.33, random_state=42)

prct_predict = o_v_o(x_train, x_test, y_train, y_test)
print(f"Predict ratio : {prct_predict:.2f}")

prct_predict = o_v_r(x_train, x_test, y_train, y_test)
print(f"Predict ratio : {prct_predict:.2f}")