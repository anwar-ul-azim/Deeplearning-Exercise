# This guide trains a neural network model to classify images of clothing, like sneakers and shirts.

#import TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)  #optional test import

# data-set
# https://github.com/zalandoresearch/fashion-mnist
fashion_mnist = keras.datasets.fashion_mnist

# Loading the dataset returns four NumPy arrays:
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each image is mapped to a single label.to use later when plotting the images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data  (optional)
# The following code shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
print(test_images.shape)
print(len(test_labels))

# Preprocess the data
# The data must be preprocessed before training the network
# show image sample
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# show image sample after Preprocess the data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Build the model
# Setup the layers
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array
# (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels
# in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are
# densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second
# (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. Each node
# contains a score that indicates the probability that the current image belongs to one of the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
# Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer"
#                the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the
#             images that are correctly classified.
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# Training the neural network model requires the following steps:
# Feed the training data to the model—in this example, the train_images and train_labels arrays.
# The model learns to associate images and labels.
# We ask the model to make predictions about a test set—in this example, the test_images array. We verify that
# the predictions match the labels from the test_labels array.
# To start training, call the model.fit method—the model is "fit" to the training data:
model.fit(train_images, train_labels, epochs=5)
# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.


# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap
# between training accuracy and test accuracy is an example of overfitting. Overfitting is when a machine learning model
# performs worse on new data than on their training data.


# Make predictions
predictions = model.predict(test_images)
# first prediction:
predictions[0]
# A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to
# each of the 10 different articles of clothing. We can see which label has the highest confidence value:
np.argmax(predictions[0])
# So the model is most confident that this image is an ankle boot, or class_names[9]. And we can check the test label to see this is correct:
test_labels[0]

# We can graph this to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Let's look at the 0th image, predictions, and prediction array.
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# Let's plot several images with their predictions. Correct prediction labels are blue and incorrect prediction labels
# are red. The number gives the percent (out of 100) for the predicted label. Note that it can be wrong even when very confident.

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Finally, use the trained model to make a prediction about a single image.
# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

# Now predict the image:
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# model.predict returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch:
np.argmax(predictions_single[0])

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.