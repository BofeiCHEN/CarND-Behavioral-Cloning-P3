
import os
import csv
import cv2
import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers



samples = []
#read csv file
data_path = '../data/behavioral-data/'
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(path, samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = path + 'IMG/'+batch_sample[0].split('/')[-1]
                left_name = './IMG/'+batch_sample[1].split('/')[-1]
                right_name = './IMG/'+batch_sample[0].split('/')[-1]
                #print(center_name)
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])

                images.append(center_image)
                angles.append(center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield tuple(map(tuple, shuffle(X_train, y_train)))

batch_size = 32

train_generator = generator(data_path, train_samples, batch_size=batch_size)
validation_generator = generator(data_path, validation_samples, batch_size=batch_size)

# image format
ch, row, col = 3, 160, 320
crop_top, crop_bottom, crop_left, crop_right = 70, 25, 0, 0

###network model
model =  keras.Sequential()
    #preprocessing data, normailize and cropping
model.add(layers.Lambda(lambda x: x / 255.0 - 0.5, input_shape = (row, col, ch)))
model.add(layers.Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right))))
model.add(layers.Conv2D(24, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(36, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(48, 5, strides=(2,2), activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dense(50))
model.add(layers.Dense(10))

model.summary()
#keras.utils.plot_model(model, "my_first_model.png")
model.compile(loss='mse', optimizer='adam')
model.fit(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

#model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)*4, nb_epoch = 2, validation_data=validation_generator, nb_val_samples=len(validation_samples))
print("END of model.py")

