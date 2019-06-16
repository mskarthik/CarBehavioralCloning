import csv
import cv2
import sklearn
import math
import random

import numpy as np
import tensorflow as tf

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

samples = []
runs = ['data','revdata','fixdata']
for data in runs:
    filename = "./%s/driving_log.csv" % data
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for line in batch_samples:
                ctr_image = cv2.imread(line[0])
                ctr_measurement = float(line[3])
                lft_image = cv2.imread(line[1])
                lft_measurement = ctr_measurement + 0.15
                rgt_image = cv2.imread(line[2])
                rgt_measurement = ctr_measurement - 0.15
                images.append(ctr_image)  
                measurements.append(ctr_measurement) 
                images.append(lft_image)
                measurements.append(lft_measurement)
                images.append(rgt_image)
                measurements.append(rgt_measurement)
                images.append(cv2.flip(ctr_image,1)) ##augmented image data
                measurements.append(ctr_measurement*(-1.0)) ##augmented measurement data       
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
# Set our batch size
batch_size=8

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) ##Normalize
model.add(Cropping2D(cropping=((70,25),(0,0))))                       ##Crop
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')

model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')

