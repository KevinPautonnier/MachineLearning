"""
    Dixième exercice de notre cours de machine learning.
    Le but était d'utiliser Keras qui est une bibliothèque basée
    sur Tenserflow pour interagir avec l'algorithmes de réseaux 
    de neurones profonds appelé MLP (Multilayer Perceptrons) et
    de le comprendre
"""

from __future__ import print_function

import timeit
import functools
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import datasets
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 20

DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

def MLP_train(x_train, y_train, x_test, y_test, model):
    #the model will output array of shape (*, 512) and the input shape is 8x8 with sklearn images
    # activation function : rectified Linear Unit (relu)
    model.add(Dense(512, activation='relu', input_shape=(64,)))
    #Dropout consists in randomly setting a fraction rate of input units to 0 
    # at each update during training time, which helps prevent overfitting.
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    #prints a summary representation of the model
    model.summary()

    #Configures the model for training.
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    return model

def MLP_test(x_test, y_test, model):
    #fit the model with training malues
    score = model.evaluate(x_test, y_test, verbose=0)

    return score

def MLP_init(x_train, x_test, y_train, y_test):
    #reshape with the number of images and their size
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #The Sequential model is a linear stack of layers
    model = Sequential()
    
    time_to_train = 0
    MLP_train(x_train, y_train, x_test, y_test, model)
    time_to_train += timeit.timeit(functools.partial(MLP_train, x_train, y_train, x_test, y_test, model), number=1)

    time_to_test = 0
    score = MLP_test(x_test, y_test, model)
    time_to_test += timeit.timeit(functools.partial(MLP_test, x_test, y_test, model), number=1)
    
    test_loss = score[0]
    test_accuracy = score[1]

    return test_accuracy*100, time_to_train, time_to_test

def main():
        
    #Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
    # the data, split between train and test sets
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Use images of sklearn instead of mnist images
    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.2)

    prct_predict, time_to_train, time_to_test = MLP_init(x_train, x_test, y_train, y_test)
    print("Multilayer Perceptrons")
    print(f"Predict ratio : {prct_predict:.2f}% Time to train : {time_to_train:.2f} Time to test : {time_to_test:.2f}")

if __name__ == "__main__":
    main()