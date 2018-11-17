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

classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units=512, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))
                    # activity_regularizer=regularizers.l1(0.01)))

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

# Found 52992 images belonging to 24 classes.
# Found 12782 images belonging to 24 classes.
# Epoch 1/30
# 414/414 [==============================] - 123s 297ms/step - loss: 1.6952 - acc: 0.6667 - val_loss: 1.6334 - val_acc: 0.6491
# Epoch 2/30
# 414/414 [==============================] - 103s 249ms/step - loss: 0.7855 - acc: 0.8830 - val_loss: 1.6282 - val_acc: 0.6701
# ..........................................
# Epoch 14/30
# 414/414 [==============================] - 100s 240ms/step - loss: 0.3643 - acc: 0.9658 - val_loss: 1.3982 - val_acc: 0.7413
# Epoch 15/30
# 414/414 [==============================] - 114s 276ms/step - loss: 0.3520 - acc: 0.9681 - val_loss: 1.4680 - val_acc: 0.7144