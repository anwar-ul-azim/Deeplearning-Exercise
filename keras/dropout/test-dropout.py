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


# ouput

# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.

# Epoch 1/30
# 414/414 [==============================] - 156s 376ms/step - loss: 1.3528 - acc: 0.5739 - val_loss: 1.1772 - val_acc: 0.7078
# Epoch 2/30
# 414/414 [==============================] - 99s 239ms/step - loss: 0.3945 - acc: 0.8750 - val_loss: 0.9635 - val_acc: 0.7542
# Epoch 3/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.2394 - acc: 0.9238 - val_loss: 1.1335 - val_acc: 0.7305
# Epoch 4/30
# 414/414 [==============================] - 99s 238ms/step - loss: 0.1679 - acc: 0.9479 - val_loss: 0.9973 - val_acc: 0.7859
# Epoch 5/30
# 414/414 [==============================] - 99s 239ms/step - loss: 0.1350 - acc: 0.9566 - val_loss: 1.0346 - val_acc: 0.7496
# Epoch 6/30
# 414/414 [==============================] - 99s 238ms/step - loss: 0.1176 - acc: 0.9616 - val_loss: 1.1416 - val_acc: 0.7599
# Epoch 7/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.1095 - acc: 0.9643 - val_loss: 0.9984 - val_acc: 0.7644
# Epoch 8/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0903 - acc: 0.9713 - val_loss: 1.1298 - val_acc: 0.7607
# Epoch 9/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0808 - acc: 0.9733 - val_loss: 1.2391 - val_acc: 0.7898
# Epoch 10/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0767 - acc: 0.9752 - val_loss: 1.1757 - val_acc: 0.7275
# Epoch 11/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0739 - acc: 0.9761 - val_loss: 1.3507 - val_acc: 0.7507
# Epoch 12/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.0670 - acc: 0.9783 - val_loss: 1.2010 - val_acc: 0.7841
# Epoch 13/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0624 - acc: 0.9802 - val_loss: 1.0863 - val_acc: 0.7950
# Epoch 14/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.0624 - acc: 0.9804 - val_loss: 1.1819 - val_acc: 0.7903
# Epoch 15/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0599 - acc: 0.9810 - val_loss: 1.3741 - val_acc: 0.7832
# Epoch 16/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0545 - acc: 0.9823 - val_loss: 1.1370 - val_acc: 0.8049
# Epoch 17/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0471 - acc: 0.9856 - val_loss: 1.2269 - val_acc: 0.7963
# Epoch 18/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0516 - acc: 0.9835 - val_loss: 1.3117 - val_acc: 0.7471
# Epoch 19/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0445 - acc: 0.9860 - val_loss: 1.1593 - val_acc: 0.7790
# Epoch 20/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0480 - acc: 0.9846 - val_loss: 1.0785 - val_acc: 0.8025
# Epoch 21/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0489 - acc: 0.9851 - val_loss: 0.9754 - val_acc: 0.8106
# Epoch 22/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0433 - acc: 0.9865 - val_loss: 1.3908 - val_acc: 0.7985
# Epoch 23/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0460 - acc: 0.9853 - val_loss: 1.1194 - val_acc: 0.7657
# Epoch 24/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0409 - acc: 0.9868 - val_loss: 1.0185 - val_acc: 0.8043
# Epoch 25/30
# 414/414 [==============================] - 97s 234ms/step - loss: 0.0470 - acc: 0.9855 - val_loss: 1.1902 - val_acc: 0.8069
# Epoch 26/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.0389 - acc: 0.9879 - val_loss: 1.3328 - val_acc: 0.8225
# Epoch 27/30
# 414/414 [==============================] - 98s 237ms/step - loss: 0.0417 - acc: 0.9869 - val_loss: 1.0373 - val_acc: 0.8061
# Epoch 28/30
# 414/414 [==============================] - 99s 239ms/step - loss: 0.0394 - acc: 0.9878 - val_loss: 1.3407 - val_acc: 0.7700
# Epoch 29/30
# 414/414 [==============================] - 98s 236ms/step - loss: 0.0385 - acc: 0.9876 - val_loss: 1.0962 - val_acc: 0.8116
# Epoch 30/30
# 414/414 [==============================] - 97s 235ms/step - loss: 0.0402 - acc: 0.9869 - val_loss: 1.3832 - val_acc: 0.8183
