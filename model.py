import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers import Convolution2D,Cropping2D
from keras.layers import MaxPooling2D,Input,AtrousConvolution2D
from keras.callbacks import TensorBoard

#dataloader
lists = pd.read_csv('./data3/driving_log.csv')
images = []
measurements = []
img_names = lists['center']
steer_names = lists['steering']

#flip data
for img_name in img_names:
    img = cv2.imread(img_name)
    images.append(img)
    img_flip = np.fliplr(img)
    images.append(img_flip)

for steer_name in steer_names:
    measurement = float(steer_name)
    measurements.append(measurement)
    measurement_flip = -measurement
    measurements.append(measurement_flip)

X_train = np.array(images)
Y_train = np.array(measurements)
assert len(X_train)==len(Y_train)

#model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=X_train.shape[1:]))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(1))

#train
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=6,callbacks=[TensorBoard(log_dir='./log')])
print(history_object.history.keys())
model.save('model.h5')

#plot
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

