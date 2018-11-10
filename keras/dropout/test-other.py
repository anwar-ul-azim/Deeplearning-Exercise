# keras layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import regularizers
# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# cnn model creating
# initialising cnn
classifier = Sequential()

# adding layers
classifier.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

# classifier.add(Convolution2D(512, (3, 3), activation='relu'))
# classifier.add(Convolution2D(512, (3, 3), activation='relu'))
# classifier.add(Convolution2D(512, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.3))

classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
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
                                    rotation_range=20,
                                    horizontal_flip=True,
                                    # width_shift_range=0.2,
                                    # height_shift_range=0.2,
                                    # fill_mode='nearest'
                                    )

test_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    # zoom_range=0.2,
                                    # horizontal_flip=True
                                    )

training_set_a = train_datagen.flow_from_directory('A',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical'
                                                    )


test_set = test_datagen.flow_from_directory('E',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical'
                                            )


# f=test_datagen.flow_from_directory('f',
#                                 target_size=(64, 64),
#                                 batch_size=128,
#                                 class_mode='categorical',
#                                 shuffle=True,
#                                 save_to_dir='g', 
#                                 save_prefix='', 
#                                 save_format='png'
#                                 )

classifier.fit_generator(training_set_a,
                        steps_per_epoch=1656,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=399,
                        shuffle=False
                        )
classifier.save_weights('model.h5')  

# 414/414 [==============================] - 102s 245ms/step - loss: 3.4775 - acc: 0.2573 - val_loss: 2.2919 - val_acc: 0.3683
# Epoch 2/30
# 414/414 [==============================] - 98s 236ms/step - loss: 1.4099 - acc: 0.5853 - val_loss: 1.9543 - val_acc: 0.5019
# Epoch 3/30
# 414/414 [==============================] - 103s 249ms/step - loss: 1.0320 - acc: 0.7066 - val_loss: 1.6494 - val_acc: 0.5682
# Epoch 4/30
# 414/414 [==============================] - 315s 761ms/step - loss: 0.8440 - acc: 0.7659 - val_loss: 1.6201 - val_acc: 0.5767
# ,kernel_regularizer=regularizers.l1(0.01)

# 414/414 [==============================] - 126s 305ms/step - loss: 1.9779 - acc: 0.3701 - val_loss: 1.3345 - val_acc: 0.6318
# Epoch 2/30
# 414/414 [==============================] - 183s 443ms/step - loss: 0.7414 - acc: 0.7622 - val_loss: 1.2104 - val_acc: 0.6821
# Epoch 3/30
# 414/414 [==============================] - 163s 395ms/step - loss: 0.4616 - acc: 0.8525 - val_loss: 1.3514 - val_acc: 0.6748
# Epoch 4/30
# 414/414 [==============================] - 183s 442ms/step - loss: 0.3244 - acc: 0.8967 - val_loss: 1.3049 - val_acc: 0.7067
# Epoch 5/30
# 414/414 [==============================] - 151s 365ms/step - loss: 0.2627 - acc: 0.9158 - val_loss: 1.1799 - val_acc: 0.7369
# Epoch 6/30
# 414/414 [==============================] - 150s 362ms/step - loss: 0.2258 - acc: 0.9279 - val_loss: 1.5320 - val_acc: 0.6991
# Epoch 7/30
# 414/414 [==============================] - 150s 363ms/step - loss: 0.1918 - acc: 0.9379 - val_loss: 1.3684 - val_acc: 0.6859
# Epoch 8/30
# 414/414 [==============================] - 151s 364ms/step - loss: 0.1751 - acc: 0.9441 - val_loss: 1.4222 - val_acc: 0.7239
# Epoch 9/30
# 414/414 [==============================] - 149s 361ms/step - loss: 0.1616 - acc: 0.9474 - val_loss: 1.3941 - val_acc: 0.7508

# 414/414 [==============================] - 138s 333ms/step - loss: 4.9210 - acc: 0.3085 - val_loss: 8.1992 - val_acc: 0.2996
# Epoch 2/30
# 414/414 [==============================] - 156s 378ms/step - loss: 10.7538 - acc: 0.6126 - val_loss: 15.2535 - val_acc: 0.4947
# Epoch 3/30
# 414/414 [==============================] - 164s 395ms/step - loss: 17.3704 - acc: 0.7188 - val_loss: 21.3025 - val_acc: 0.4839
# Epoch 4/30
# 414/414 [==============================] - 309s 746ms/step - loss: 23.1366 - acc: 0.7534 - val_loss: 26.9121 - val_acc: 0.6023
# Epoch 5/30
# 414/414 [==============================] - 291s 703ms/step - loss: 28.1043 - acc: 0.7672 - val_loss: 30.8339 - val_acc: 0.5827



# classifier.add(Convolution2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64, 64, 3)))
# classifier.add(Convolution2D(64, kernel_size=4, strides=2, activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Convolution2D(128, kernel_size=4, strides=1, activation='relu'))
# classifier.add(Convolution2D(128, kernel_size=4, strides=2, activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Convolution2D(256, kernel_size=4, strides=1, activation='relu'))
# classifier.add(Convolution2D(256, kernel_size=4, strides=2, activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Flatten())
# classifier.add(Dense(512, activation='relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Dense(units=24, activation='softmax'))

# # output
# (keras) C:\Users\anwar\gitkraken\Deeplearning Exercise\keras\dropout>python test-other.py
# Using TensorFlow backend.
# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/30
# 2018-11-10 22:24:33.582526: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# 2018-11-10 22:24:34.051824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties:
# name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.455
# pciBusID: 0000:09:00.0
# totalMemory: 2.00GiB freeMemory: 1.60GiB
# 2018-11-10 22:24:34.060626: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
# 2018-11-10 22:24:34.949343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-11-10 22:24:34.952411: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0
# 2018-11-10 22:24:34.954271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N
# 2018-11-10 22:24:34.956224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1351 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:09:00.0, compute capability: 6.1)
# 414/414 [==============================] - 101s 243ms/step - loss: 2.2179 - acc: 0.2841 - val_loss: 1.6590 - val_acc: 0.5037
# Epoch 2/30
# 414/414 [==============================] - 98s 236ms/step - loss: 1.0559 - acc: 0.6513 - val_loss: 1.3728 - val_acc: 0.6218
# Epoch 3/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.6659 - acc: 0.7835 - val_loss: 1.4571 - val_acc: 0.6737
# Epoch 4/30
# 414/414 [==============================] - 96s 232ms/step - loss: 0.4903 - acc: 0.8411 - val_loss: 1.3564 - val_acc: 0.6496
# Epoch 5/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.3856 - acc: 0.8743 - val_loss: 1.2306 - val_acc: 0.6910
# Epoch 6/30
# 414/414 [==============================] - 97s 233ms/step - loss: 0.3395 - acc: 0.8894 - val_loss: 1.3292 - val_acc: 0.6997
# Epoch 7/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.2865 - acc: 0.9073 - val_loss: 1.1668 - val_acc: 0.7119
# Epoch 8/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.2624 - acc: 0.9160 - val_loss: 1.1462 - val_acc: 0.7032
# Epoch 9/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.2351 - acc: 0.9225 - val_loss: 1.1754 - val_acc: 0.7433
# Epoch 10/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.2127 - acc: 0.9321 - val_loss: 1.2587 - val_acc: 0.7037
# Epoch 11/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.2021 - acc: 0.9345 - val_loss: 1.1347 - val_acc: 0.7288
# Epoch 12/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1958 - acc: 0.9372 - val_loss: 1.1469 - val_acc: 0.7492
# Epoch 13/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1823 - acc: 0.9412 - val_loss: 1.2034 - val_acc: 0.7420
# Epoch 14/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1743 - acc: 0.9440 - val_loss: 1.2769 - val_acc: 0.7415
# Epoch 15/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1621 - acc: 0.9482 - val_loss: 1.1063 - val_acc: 0.7212
# Epoch 16/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1610 - acc: 0.9486 - val_loss: 1.1917 - val_acc: 0.6980
# Epoch 17/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1518 - acc: 0.9508 - val_loss: 1.1655 - val_acc: 0.7606
# Epoch 18/30
# 414/414 [==============================] - 96s 232ms/step - loss: 0.1479 - acc: 0.9537 - val_loss: 1.1974 - val_acc: 0.7482
# Epoch 19/30
# 414/414 [==============================] - 96s 233ms/step - loss: 0.1510 - acc: 0.9533 - val_loss: 1.4173 - val_acc: 0.7268
# Epoch 20/30
# 414/414 [==============================] - 97s 233ms/step - loss: 0.1356 - acc: 0.9572 - val_loss: 1.3155 - val_acc: 0.7248
# Epoch 21/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.1428 - acc: 0.9548 - val_loss: 1.0661 - val_acc: 0.7335
# Epoch 22/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.1324 - acc: 0.9591 - val_loss: 1.3276 - val_acc: 0.7111
# Epoch 23/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.1282 - acc: 0.9604 - val_loss: 1.1871 - val_acc: 0.7076
# Epoch 24/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.1266 - acc: 0.9599 - val_loss: 1.2393 - val_acc: 0.7376
# Epoch 25/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.1223 - acc: 0.9603 - val_loss: 1.1991 - val_acc: 0.7495
# Epoch 26/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.1276 - acc: 0.9601 - val_loss: 1.1176 - val_acc: 0.7742
# Epoch 27/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.1196 - acc: 0.9625 - val_loss: 1.2061 - val_acc: 0.7508
# Epoch 28/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.1170 - acc: 0.9632 - val_loss: 1.2066 - val_acc: 0.7647
# Epoch 29/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.1284 - acc: 0.9599 - val_loss: 1.1428 - val_acc: 0.7215
# Epoch 30/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.1134 - acc: 0.9654 - val_loss: 1.1155 - val_acc: 0.7571




# kernel 4


# 414/414 [==============================] - 132s 319ms/step - loss: 1.8288 - acc: 0.4098 - val_loss: 1.4667 - val_acc: 0.5800
# Epoch 2/30
# 414/414 [==============================] - 115s 278ms/step - loss: 0.6857 - acc: 0.7780 - val_loss: 1.2469 - val_acc: 0.6760
# Epoch 3/30
# 414/414 [==============================] - 109s 264ms/step - loss: 0.3889 - acc: 0.8737 - val_loss: 1.0158 - val_acc: 0.7359
# Epoch 4/30
# 414/414 [==============================] - 109s 263ms/step - loss: 0.2657 - acc: 0.9140 - val_loss: 1.3427 - val_acc: 0.7254
# Epoch 5/30
# 414/414 [==============================] - 110s 265ms/step - loss: 0.2146 - acc: 0.9296 - val_loss: 1.1388 - val_acc: 0.7621
# Epoch 6/30
# 414/414 [==============================] - 109s 263ms/step - loss: 0.1780 - acc: 0.9441 - val_loss: 0.8943 - val_acc: 0.7940
# Epoch 7/30
# 414/414 [==============================] - 109s 264ms/step - loss: 0.1584 - acc: 0.9487 - val_loss: 1.4190 - val_acc: 0.7373
# Epoch 8/30
# 414/414 [==============================] - 109s 263ms/step - loss: 0.1355 - acc: 0.9558 - val_loss: 0.9903 - val_acc: 0.7485
# Epoch 9/30
# 414/414 [==============================] - 109s 263ms/step - loss: 0.1272 - acc: 0.9592 - val_loss: 0.7636 - val_acc: 0.8013
# Epoch 10/30
# 414/414 [==============================] - 109s 264ms/step - loss: 0.1207 - acc: 0.9621 - val_loss: 1.0548 - val_acc: 0.7974
# Epoch 11/30
# 414/414 [==============================] - 109s 263ms/step - loss: 0.1095 - acc: 0.9645 - val_loss: 0.7735 - val_acc: 0.8009
# Epoch 12/30
# 414/414 [==============================] - 109s 264ms/step - loss: 0.1095 - acc: 0.9659 - val_loss: 1.0797 - val_acc: 0.8026
# Epoch 13/30
# 414/414 [==============================] - 144s 349ms/step - loss: 0.0955 - acc: 0.9695 - val_loss: 1.2330 - val_acc: 0.7624
# Epoch 14/30
# 414/414 [==============================] - 148s 358ms/step - loss: 0.0940 - acc: 0.9704 - val_loss: 1.5467 - val_acc: 0.7445
# Epoch 15/30
# 414/414 [==============================] - 150s 361ms/step - loss: 0.0926 - acc: 0.9708 - val_loss: 1.4383 - val_acc: 0.7727
# Epoch 16/30
# 414/414 [==============================] - 149s 361ms/step - loss: 0.0895 - acc: 0.9727 - val_loss: 1.2518 - val_acc: 0.7678
# Epoch 17/30
# 414/414 [==============================] - 148s 358ms/step - loss: 0.0853 - acc: 0.9734 - val_loss: 0.9587 - val_acc: 0.8010
# Epoch 18/30
# 414/414 [==============================] - 149s 359ms/step - loss: 0.0787 - acc: 0.9756 - val_loss: 0.8573 - val_acc: 0.8054
# Epoch 19/30
# 414/414 [==============================] - 149s 359ms/step - loss: 0.0814 - acc: 0.9739 - val_loss: 1.1660 - val_acc: 0.7591
# Epoch 20/30
# 414/414 [==============================] - 149s 360ms/step - loss: 0.0820 - acc: 0.9746 - val_loss: 0.9152 - val_acc: 0.8217
# Epoch 21/30
# 414/414 [==============================] - 148s 357ms/step - loss: 0.0800 - acc: 0.9755 - val_loss: 1.0681 - val_acc: 0.7703
# Epoch 22/30
# 414/414 [==============================] - 149s 359ms/step - loss: 0.0755 - acc: 0.9765 - val_loss: 1.0904 - val_acc: 0.7800
# Epoch 23/30
# 414/414 [==============================] - 100s 240ms/step - loss: 0.0714 - acc: 0.9781 - val_loss: 1.0035 - val_acc: 0.7905
# Epoch 24/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0703 - acc: 0.9792 - val_loss: 1.2962 - val_acc: 0.7391
# Epoch 25/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0752 - acc: 0.9772 - val_loss: 0.9730 - val_acc: 0.7771
# Epoch 26/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0740 - acc: 0.9769 - val_loss: 1.0897 - val_acc: 0.7950
# Epoch 27/30
# 414/414 [==============================] - 98s 238ms/step - loss: 0.0697 - acc: 0.9790 - val_loss: 1.6629 - val_acc: 0.7240
# Epoch 28/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0658 - acc: 0.9805 - val_loss: 1.0217 - val_acc: 0.7858
# Epoch 29/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0646 - acc: 0.9808 - val_loss: 1.0133 - val_acc: 0.7730
# Epoch 30/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.0647 - acc: 0.9804 - val_loss: 1.1267 - val_acc: 0.7689