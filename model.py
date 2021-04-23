import os
import csv
import cv2
import numpy as np
import tensorflow as tf
import keras
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import layers
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot



samples = []
#read csv file
data_path = '../../../opt/data/'
#data_path = 'data/'
with open(data_path + 'driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

### The generator for train data and validation data
def generator(path, samples, batch_size=32):
    correction = 0.2 # this is a parameter to tune
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = path + 'IMG/'+batch_sample[0].split('/')[-1]
                left_name = path + './IMG/'+batch_sample[1].split('/')[-1]
                right_name = path + './IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                left_image = cv2.imread(left_name)
                left_angle = center_angle + correction
                right_image = cv2.imread(right_name)
                right_angle = center_angle - correction

                # add center image
                images.append(center_image)
                angles.append(center_angle)
                
                # add fliped center image
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)
                
                # add left camera image
                images.append(left_image)
                angles.append(left_angle)
                
                # add right camera image
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield tuple( shuffle(X_train, y_train))

            
batch_size = 32

train_generator = generator(data_path, train_samples, batch_size=batch_size)
validation_generator = generator(data_path, validation_samples, batch_size=batch_size)

# image format
ch, row, col = 3, 160, 320
crop_top, crop_bottom, crop_left, crop_right = 70, 25, 0, 0

###network model
model =  keras.Sequential()
model.add(layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape = (row, col, ch))) # normailize
model.add(layers.Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)))) # cropping
model.add(layers.Conv2D(24, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(36, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(48, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dropout(.7))
model.add(layers.Dense(50))
model.add(layers.Dense(10))
model.add(layers.Dense(1))

model.summary()
keras.utils.plot_model(model, "behavioral_clone_model.png", show_shapes=True)
model.compile(loss='mse', optimizer='adam')


# simple early stopping callback functions
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# train model
history = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=10, verbose=1, callbacks=[es, mc])

model.save('model.h5')

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("losses.png")

print("END of training")

