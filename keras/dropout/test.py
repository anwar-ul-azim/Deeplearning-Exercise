# importing keras libraries
# keras layers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# cnn model creating
# initialising cnn
classifier = Sequential()

# adding layers
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Flatten())
classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dropout(0.3))
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


#model 1

# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dense(units=512, activation='relu'))
# classifier.add(Dense(units=24, activation='softmax'))

# output 1

# (keras) G:\asl a dataset\dataset5>python test.py
# Using TensorFlow backend.
# Found 106080 images belonging to 24 classes.
# Found 25588 images belonging to 24 classes.
# Epoch 1/15
# 414/414 [==============================] - 1841s 4s/step - loss: 1.2353 - acc: 0.6240 - val_loss: 1.8102 - val_acc: 0.5441
# Epoch 2/15
# 414/414 [==============================] - 631s 2s/step - loss: 0.5108 - acc: 0.8379 - val_loss: 1.8146 - val_acc: 0.5596
# .................
# Epoch 14/15
# 414/414 [==============================] - 184s 445ms/step - loss: 0.1309 - acc: 0.9566 - val_loss: 2.2141 - val_acc: 0.6212
# Epoch 15/15
# 414/414 [==============================] - 185s 447ms/step - loss: 0.1237 - acc: 0.9584 - val_loss: 1.9271 - val_acc: 0.6482

#output-2

# (keras) G:\asl a dataset\dataset5>python test.py
# Using TensorFlow backend.
# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/15
# 414/414 [==============================] - 1174s 3s/step - loss: 0.7683 - acc: 0.7673 - val_loss: 1.2332 - val_acc: 0.7079
# Epoch 2/15
# 414/414 [==============================] - 190s 459ms/step - loss: 0.1291 - acc: 0.9595 - val_loss: 1.4044 - val_acc: 0.6549
#...................................
# Epoch 14/15
# 414/414 [==============================] - 193s 466ms/step - loss: 0.0111 - acc: 0.9964 - val_loss: 1.5456 - val_acc: 0.7547
# Epoch 15/15
# 414/414 [==============================] - 197s 477ms/step - loss: 0.0109 - acc: 0.9966 - val_loss: 1.6994 - val_acc: 0.7376

# model 2

# classifier.add(Convolution2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Convolution2D(256, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(256, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dense(units=512, activation='relu'))
# classifier.add(Dense(units=24, activation='softmax'))

# output 3

# (keras) G:\asl a dataset\dataset5>python test.py
# Using TensorFlow backend.
# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/15
# 414/414 [==============================] - 677s 2s/step - loss: 1.0221 - acc: 0.6738 - val_loss: 1.5382 - val_acc: 0.6938
# Epoch 2/15
# 414/414 [==============================] - 305s 736ms/step - loss: 0.1872 - acc: 0.9404 - val_loss: 1.6346 - val_acc: 0.6980
#...........................................
# Epoch 14/15
# 414/414 [==============================] - 199s 480ms/step - loss: 0.0198 - acc: 0.9936 - val_loss: 1.6406 - val_acc: 0.7436
# Epoch 15/15
# 414/414 [==============================] - 198s 477ms/step - loss: 0.0194 - acc: 0.9938 - val_loss: 1.6513 - val_acc: 0.7760

#model 3

# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(Dropout(0.5))

# classifier.add(Convolution2D(128, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# classifier.add(Flatten())
# classifier.add(Dropout(0.5))

# classifier.add(Dense(units=512, activation='relu'))
# classifier.add(Dense(units=24, activation='softmax'))

# output 4

# (keras) G:\asl a dataset\dataset5>python test.py
# Using TensorFlow backend.
# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/15
## 414/414 [==============================] - 234s 566ms/step - loss: 1.0682 - acc: 0.6724 - val_loss: 1.2903 - val_acc: 0.6835
# Epoch 2/15
# 414/414 [==============================] - 193s 465ms/step - loss: 0.1913 - acc: 0.9387 - val_loss: 1.3160 - val_acc: 0.7047
#...........................................................................
#  Epoch 14/15
# 414/414 [==============================] - 192s 464ms/step - loss: 0.0245 - acc: 0.9920 - val_loss: 1.3613 - val_acc: 0.7507
# Epoch 15/15
# 414/414 [==============================] - 192s 464ms/step - loss: 0.0217 - acc: 0.9928 - val_loss: 1.2561 - val_acc: 0.7743