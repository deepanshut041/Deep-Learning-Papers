#!/usr/bin/env python
# coding: utf-8

# # AlexNet

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Data Loading and Preprocessing

# ### CIFAR10 small image classification
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
# 
# #### Returns 2 tuples:
# - **x_train, x_test**: uint8 array of RGB image data with shape (num_samples, 32, 32, 3).
# - **y_train, y_test**: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).


from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
print("Number of training examples =", len(x_train))
print("Number of testing examples =", len(x_test))
print("Image data shape =", x_train[0].shape)
print("Number of classes =", num_classes)


# convert to one hot encoing 
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# ## Model Implementation

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Activation, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model

# ### Model Architecture

def create_model():
    # Input Layer
    inputs = Input(x_train[0].shape)
    
    # Layer 1
    layer_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(inputs)
    
    # Layer 2
    layer_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    
    # Layer 3
    layer_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(layer_2)
    
    # Layer 4
    layer_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_3)
    layer_4 = BatchNormalization()(layer_4)
    
    # Layer 5
    layer_5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(layer_4)
    
    # Layer 6
    layer_6 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_5)
    layer_6 = BatchNormalization()(layer_6)
    
    # Layer 7
    layer_7 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(layer_6)
    
    # Layer 8
    layer_8 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer_7)
    layer_8 = BatchNormalization()(layer_8)
    
    # Layer 9
    layer_9 = Flatten()(layer_8)
    
    # Layer 10
    layer_10 = Dense(units=128, activation='relu')(layer_9)
    layer_10 = Dropout(rate=0.3)(layer_10)
    layer_10 = BatchNormalization()(layer_10)
    
    # Layer 11
    layer_11 = Dense(units=256, activation='relu')(layer_10)
    layer_11 = Dropout(rate=0.3)(layer_11)
    layer_11 = BatchNormalization()(layer_11)
    
    # Layer 12
    layer_12 = Dense(units=512, activation='relu')(layer_11)
    layer_12 = Dropout(rate=0.3)(layer_12)
    layer_12 = BatchNormalization()(layer_12)
    
    # Layer 13
    layer_13 = Dense(units=1024, activation='relu')(layer_12)
    layer_13 = Dropout(rate=0.3)(layer_13)
    layer_13 = BatchNormalization()(layer_13)
    
    # Layer 14
    layer_14 = Dense(units=10, activation='softmax')(layer_13)
    
    model = Model([inputs], layer_14)
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


model = create_model()
print(model.summary())


# ## Training
print("------------ Traning----------------------")
model.fit(x=x_train, y=y_train, epochs=1, batch_size=512, validation_split=0.2)

print("------------ Testing----------------------")
# ## Testing

print(model.evaluate(x=x_test, y=y_test))

model.save('model.h5')
history = model.history

# ## Results

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
