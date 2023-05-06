# ## Importing Dependencies:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import numpy as np
from matplotlib import pyplot as plt
import random


# ## Loading Dataset 
# (here, we have used tensorflow mnist dataset, Handwritten digits and their labels)
data = tf.keras.datasets.mnist

#unpacking data:
#    images to x_train/x_test
#    labels to y_train/y_test
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
print(x_train.shape)
print(x_train[0].shape)


fig = plt.figure(figsize=(5, 5))
rows = 2
columns = 3

fig.add_subplot(rows, columns, 1)
plt.imshow(x_train[0])
plt.title(y_train[0])
plt.axis('off')

fig.add_subplot(rows, columns, 2)
plt.imshow(x_train[3])
plt.title(y_train[3])
plt.axis('off')

fig.add_subplot(rows, columns, 3)
plt.imshow(x_train[90])
plt.title(y_train[90])
plt.axis('off')

fig.add_subplot(rows, columns, 4)
plt.imshow(x_train[5])
plt.title(y_train[5])
plt.axis('off')

fig.add_subplot(rows, columns, 5)
plt.imshow(x_train[10])
plt.title(y_train[10])
plt.axis('off')

fig.add_subplot(rows, columns, 6)
plt.imshow(x_train[300])
plt.title(y_train[300])
plt.axis('off')


print("(28 x 28 pixels per image)")


# ## Model Formation:
# Loading Sequential Model for Layer Stacking
model = Sequential()
print(x_train.shape[1:])

# Adding Layers to loaded model as per requirement
# Here we will use LSTM and Dense
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, 
# which helps prevent overfitting
model.add(LSTM(128, activation="relu"))
model.add(Dropout(0.1))
# Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer. 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

# Setting up an Optimizer
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

# Compiling the model (Building up Neural Network)
model.compile(
loss='sparse_categorical_crossentropy',
optimizer=opt,
metrics=['accuracy']
)

model.summary()


# ## Training the model:
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# ## Prediction:
y_predict = model.predict(x_test)
print(y_predict)
print(y_predict.shape)

fig = plt.figure(figsize=(5, 5))
rows = 4
columns = 4

index = random.randint(0,9999)

fig.add_subplot(rows, columns, 1)
plt.imshow(x_test[index])
plt.title("Predicted Value: {}\nOriginal Input:".format(y_predict[index].argmax()))
plt.axis('off')
print("predicted value: {}".format(y_predict[index].argmax()))
print("Original test value: {}".format(y_test[index]))
