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
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# extra layer
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
############
classifier.add(Flatten())
classifier.add(Dense(output_dim=256, activation='relu'))
classifier.add(Dense(output_dim=29, activation='softmax'))

# compiling cnn
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy']
                   )

# cnn model training & testing
# data-set
# link: https://www.kaggle.com/danrasband/classifying-images-of-the-asl-alphabet-using-keras/data
# The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which
#           represent the various classes.
# The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the
#           letters A-Z and 3 classes for SPACE, DELETE and NOTHING
# And The test data set contains 870 images in 29 classes
# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical'
                                                 )

test_set = test_datagen.flow_from_directory('data/test_set',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='categorical'
                                            )

classifier.fit_generator(training_set,
                         samples_per_epoch=87000,
                         epochs=25,
                         validation_data=test_set,
                         nb_val_samples=870
                         )
