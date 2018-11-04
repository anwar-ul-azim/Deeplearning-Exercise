# importing keras libraries
# keras layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# cnn model creating
# initialising cnn
classifier = Sequential()

# adding layers
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# extra layer
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
############
classifier.add(Flatten())
classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

# compiling cnn
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy']
                   )

# cnn model training & testing
# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set_a = train_datagen.flow_from_directory('A',
                                                 target_size=(64, 64),
                                                 batch_size=256,
                                                 class_mode='categorical'
                                                 )


test_set = test_datagen.flow_from_directory('E',
                                            target_size=(64, 64),
                                            batch_size=256,
                                            class_mode='categorical'
                                            )

classifier.fit_generator(training_set_a,
                         steps_per_epoch=414,
                         epochs=15,
                         validation_data=test_set,
                         validation_steps=99,
                         shuffle=False
                         )

# output

# (keras) G:\asl a dataset\dataset5>python test.py
# Using TensorFlow backend.
# Found 106080 images belonging to 24 classes.
# Found 25588 images belonging to 24 classes.

# 2018-11-04 21:40:05.139954: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this 
# TensorFlow binary was not compiled to use: AVX22018-11-04 21:40:05.590572: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] 
# Found device 0 with properties:
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.455pciBusID: 0000:09:00.0
# totalMemory: 2.00GiB freeMemory: 1.60GiB2018-11-04 21:40:05.596095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
# 2018-11-04 21:40:06.458293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-11-04 21:40:06.462064: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      02018-11-04 21:40:06.464146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:  
#  N2018-11-04 21:40:06.466298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1351 MB memory) -> 
# physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:09:00.0, compumemory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:09:00.0, compute capability: 6.1)
# Epoch 1/15
# 414/414 [==============================] - 1841s 4s/step - loss: 1.2353 - acc: 0.6240 - val_loss: 1.8102 - val_acc: 0.5441
# Epoch 2/15
# 414/414 [==============================] - 631s 2s/step - loss: 0.5108 - acc: 0.8379 - val_loss: 1.8146 - val_acc: 0.5596
# Epoch 3/15
# 414/414 [==============================] - 382s 922ms/step - loss: 0.3657 - acc: 0.8828 - val_loss: 1.6181 - val_acc: 0.6077
# Epoch 4/15
# 414/414 [==============================] - 180s 434ms/step - loss: 0.2911 - acc: 0.9058 - val_loss: 1.7335 - val_acc: 0.5991
# Epoch 5/15
# 414/414 [==============================] - 190s 458ms/step - loss: 0.2551 - acc: 0.9172 - val_loss: 1.6405 - val_acc: 0.6073
# Epoch 6/15
# 414/414 [==============================] - 238s 574ms/step - loss: 0.2228 - acc: 0.9263 - val_loss: 1.6976 - val_acc: 0.6107
# Epoch 7/15
# 414/414 [==============================] - 188s 455ms/step - loss: 0.2033 - acc: 0.9332 - val_loss: 1.8214 - val_acc: 0.6121
# Epoch 8/15
# 414/414 [==============================] - 222s 536ms/step - loss: 0.1852 - acc: 0.9390 - val_loss: 1.8857 - val_acc: 0.6129
# Epoch 9/15
# 414/414 [==============================] - 191s 461ms/step - loss: 0.1734 - acc: 0.9423 - val_loss: 1.8031 - val_acc: 0.6292
# Epoch 10/15
# 414/414 [==============================] - 186s 450ms/step - loss: 0.1613 - acc: 0.9460 - val_loss: 1.7825 - val_acc: 0.6224
# Epoch 11/15
# 414/414 [==============================] - 185s 448ms/step - loss: 0.1506 - acc: 0.9501 - val_loss: 1.7732 - val_acc: 0.6384
# Epoch 12/15
# 414/414 [==============================] - 186s 449ms/step - loss: 0.1432 - acc: 0.9524 - val_loss: 2.0312 - val_acc: 0.6187
# Epoch 13/15
# 414/414 [==============================] - 188s 454ms/step - loss: 0.1369 - acc: 0.9550 - val_loss: 2.0793 - val_acc: 0.6192
# Epoch 14/15
# 414/414 [==============================] - 184s 445ms/step - loss: 0.1309 - acc: 0.9566 - val_loss: 2.2141 - val_acc: 0.6212
# Epoch 15/15
# 414/414 [==============================] - 185s 447ms/step - loss: 0.1237 - acc: 0.9584 - val_loss: 1.9271 - val_acc: 0.6482