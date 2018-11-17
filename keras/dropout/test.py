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
                        