"""
    Dixième exercice de notre cours de machine learning.
    Le but était d'utiliser Keras qui est une bibliothèque basée
    sur Tenserflow pour interagir avec l'algorithmes de réseaux 
    de neurones profonds appelé MLP (Multilayer Perceptrons) et
    de le comprendre
"""

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import datasets
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 20

#Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshape with the number of images and their size
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#The Sequential model is a linear stack of layers
model = Sequential()
#the model will output array of shape (*, 512) and the input shape is (*, 784)
# activation function : rectified Linear Unit (relu)
model.add(Dense(512, activation='relu', input_shape=(784,)))
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

#fit the model with training malues
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

#calculate the score of the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])