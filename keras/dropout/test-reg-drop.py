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
classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l1(0.005)))
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
                                                    batch_size=128,
                                                    class_mode='categorical'
                                                    )


test_set = test_datagen.flow_from_directory('E',
                                            target_size=(64, 64),
                                            batch_size=128,
                                            class_mode='categorical'
                                            )

classifier.fit_generator(training_set_a,
                        steps_per_epoch=414,
                        epochs=30,
                        validation_data=test_set,
                        validation_steps=99,
                        shuffle=False
                        )



# ? model 1

# classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(Dropout(0.2))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.2))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l1(0.01)))
# classifier.add(Dense(units=24, activation='softmax'))

# Epoch 1/30
# 414/414 [==============================] - 101s 245ms/step - loss: 21.6177 - acc: 0.3833 - val_loss: 7.0823 - val_acc: 0.4692
# Epoch 2/30
# 414/414 [==============================] - 97s 234ms/step - loss: 5.9604 - acc: 0.6611 - val_loss: 6.3675 - val_acc: 0.5811
# Epoch 3/30
# 414/414 [==============================] - 97s 234ms/step - loss: 5.4447 - acc: 0.7450 - val_loss: 5.9574 - val_acc: 0.6003
# ..........................
# Epoch 7/30
# 414/414 [==============================] - 97s 234ms/step - loss: 4.7319 - acc: 0.8465 - val_loss: 5.1947 - val_acc: 0.6984
# Epoch 8/30
# 414/414 [==============================] - 99s 240ms/step - loss: 4.6597 - acc: 0.8594 - val_loss: 5.5832 - val_acc: 0.6436
# Epoch 9/30
# 414/414 [==============================] - 98s 237ms/step - loss: 4.5760 - acc: 0.8709 - val_loss: 5.8303 - val_acc: 0.6025
# Epoch 10/30
# 414/414 [==============================] - 97s 235ms/step - loss: 4.5536 - acc: 0.8761 - val_loss: 5.7303 - val_acc: 0.6309


# ? model 2

# # adding layers
# classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(Dropout(0.3))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.3))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.3))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l1(0.01)))
# classifier.add(Dense(units=24, activation='softmax'))

# 414/414 [==============================] - 101s 245ms/step - loss: 21.5487 - acc: 0.3836 - val_loss: 7.0100 - val_acc: 0.4022
# Epoch 2/30
# 414/414 [==============================] - 97s 233ms/step - loss: 6.1918 - acc: 0.6271 - val_loss: 6.5868 - val_acc: 0.5077
# Epoch 3/30
# 414/414 [==============================] - 96s 232ms/step - loss: 5.7177 - acc: 0.7221 - val_loss: 6.2039 - val_acc: 0.5773
# Epoch 4/30
# 414/414 [==============================] - 97s 234ms/step - loss: 5.3933 - acc: 0.7637 - val_loss: 6.0994 - val_acc: 0.5655
# Epoch 5/30
# 414/414 [==============================] - 96s 232ms/step - loss: 5.1728 - acc: 0.7924 - val_loss: 5.9793 - val_acc: 0.5919
# Epoch 6/30
# 414/414 [==============================] - 97s 235ms/step - loss: 5.0391 - acc: 0.8100 - val_loss: 6.1327 - val_acc: 0.5796
# Epoch 7/30
# 414/414 [==============================] - 96s 232ms/step - loss: 4.9636 - acc: 0.8246 - val_loss: 5.6753 - val_acc: 0.6174
# Epoch 8/30
# 414/414 [==============================] - 96s 232ms/step - loss: 4.9061 - acc: 0.8332 - val_loss: 5.6103 - val_acc: 0.6534
# Epoch 9/30
# 414/414 [==============================] - 96s 232ms/step - loss: 4.8363 - acc: 0.8445 - val_loss: 5.6668 - val_acc: 0.6375
# Epoch 10/30
# 414/414 [==============================] - 96s 233ms/step - loss: 4.7830 - acc: 0.8535 - val_loss: 5.6407 - val_acc: 0.6513



# ? model 3
# classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l2(0.1)))


# 414/414 [==============================] - 107s 258ms/step - loss: 4.1330 - acc: 0.5784 - val_loss: 2.5570 - val_acc: 0.5663
# Epoch 2/30
# 414/414 [==============================] - 101s 243ms/step - loss: 1.4905 - acc: 0.7832 - val_loss: 2.5789 - val_acc: 0.5684
# Epoch 3/30
# 414/414 [==============================] - 98s 236ms/step - loss: 1.2628 - acc: 0.8255 - val_loss: 2.0250 - val_acc: 0.6039
# Epoch 4/30
# 414/414 [==============================] - 98s 236ms/step - loss: 1.0993 - acc: 0.8498 - val_loss: 2.2615 - val_acc: 0.6219
# Epoch 5/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.9897 - acc: 0.8633 - val_loss: 2.1231 - val_acc: 0.6228
# Epoch 6/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.8979 - acc: 0.8752 - val_loss: 1.7153 - val_acc: 0.6808
# Epoch 7/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.8269 - acc: 0.8846 - val_loss: 1.7096 - val_acc: 0.6701
# Epoch 8/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.7742 - acc: 0.8892 - val_loss: 1.8878 - val_acc: 0.6375
# Epoch 9/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.7329 - acc: 0.8973 - val_loss: 1.6945 - val_acc: 0.7097
# Epoch 10/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.6894 - acc: 0.9011 - val_loss: 1.6769 - val_acc: 0.6682
# Epoch 11/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.6555 - acc: 0.9089 - val_loss: 1.9553 - val_acc: 0.5966
# Epoch 12/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.6368 - acc: 0.9098 - val_loss: 1.7144 - val_acc: 0.6733
# Epoch 13/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.5938 - acc: 0.9144 - val_loss: 2.0514 - val_acc: 0.5333
# Epoch 14/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.5767 - acc: 0.9186 - val_loss: 1.5465 - val_acc: 0.7115
# Epoch 15/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.5542 - acc: 0.9205 - val_loss: 1.3770 - val_acc: 0.7228
# Epoch 16/30
# 414/414 [==============================] - 102s 247ms/step - loss: 0.5368 - acc: 0.9235 - val_loss: 1.7897 - val_acc: 0.6248
# Epoch 17/30
# 414/414 [==============================] - 98s 238ms/step - loss: 0.5241 - acc: 0.9266 - val_loss: 1.4804 - val_acc: 0.7055
# Epoch 18/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.5104 - acc: 0.9296 - val_loss: 1.4277 - val_acc: 0.6787
# Epoch 19/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.4980 - acc: 0.9303 - val_loss: 1.4614 - val_acc: 0.7011
# Epoch 20/30
# 414/414 [==============================] - 99s 240ms/step - loss: 0.4822 - acc: 0.9347 - val_loss: 1.4664 - val_acc: 0.7104
# Epoch 21/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.4650 - acc: 0.9360 - val_loss: 1.5265 - val_acc: 0.7076
# Epoch 22/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.4558 - acc: 0.9374 - val_loss: 1.3918 - val_acc: 0.7133
# Epoch 23/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.4385 - acc: 0.9382 - val_loss: 1.3761 - val_acc: 0.7045


#  ? model 4
# classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l1(0.005)))
# classifier.add(Dense(units=24, activation='softmax'))

# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/30
# 414/414 [==============================] - 103s 249ms/step - loss: 12.0112 - acc: 0.3684 - val_loss: 4.7603 - val_acc: 0.4412
# Epoch 2/30
# 414/414 [==============================] - 99s 239ms/step - loss: 3.9377 - acc: 0.6472 - val_loss: 4.6166 - val_acc: 0.5044
# Epoch 3/30
# 414/414 [==============================] - 99s 238ms/step - loss: 3.5486 - acc: 0.7346 - val_loss: 3.9858 - val_acc: 0.6132
# Epoch 4/30
# 414/414 [==============================] - 98s 238ms/step - loss: 3.2942 - acc: 0.7721 - val_loss: 3.8186 - val_acc: 0.6110
# Epoch 5/30
# 414/414 [==============================] - 98s 237ms/step - loss: 3.1434 - acc: 0.7953 - val_loss: 3.9246 - val_acc: 0.6068
# Epoch 6/30
# 414/414 [==============================] - 99s 238ms/step - loss: 3.0507 - acc: 0.8080 - val_loss: 3.6409 - val_acc: 0.6518
# Epoch 7/30
# 414/414 [==============================] - 100s 241ms/step - loss: 2.9833 - acc: 0.8183 - val_loss: 3.7962 - val_acc: 0.6281
# Epoch 8/30
# 414/414 [==============================] - 98s 237ms/step - loss: 2.9203 - acc: 0.8319 - val_loss: 3.7287 - val_acc: 0.6296
# Epoch 9/30
# 414/414 [==============================] - 99s 238ms/step - loss: 2.8781 - acc: 0.8365 - val_loss: 3.7189 - val_acc: 0.6291
# Epoch 10/30
# 414/414 [==============================] - 98s 238ms/step - loss: 2.8423 - acc: 0.8426 - val_loss: 3.5430 - val_acc: 0.6581

# ? model 4
# # classifier.add(Dense(units=512, activation='relu',kernel_regularizer=regularizers.l2(0.005)))
# 414/414 [==============================] - 103s 250ms/step - loss: 1.7109 - acc: 0.6359 - val_loss: 1.9092 - val_acc: 0.6506
# Epoch 2/30
# 414/414 [==============================] - 102s 245ms/step - loss: 0.8690 - acc: 0.8809 - val_loss: 1.9716 - val_acc: 0.6704
# Epoch 3/30
# 414/414 [==============================] - 101s 243ms/step - loss: 0.7469 - acc: 0.9127 - val_loss: 1.8832 - val_acc: 0.6705
# Epoch 4/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.6873 - acc: 0.9236 - val_loss: 1.9424 - val_acc: 0.6507
# Epoch 5/30
# 414/414 [==============================] - 104s 251ms/step - loss: 0.6437 - acc: 0.9337 - val_loss: 1.6688 - val_acc: 0.7017