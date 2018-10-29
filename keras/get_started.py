# Get Started with TensorFlow
# https://www.tensorflow.org/tutorials/
import tensorflow as tf

# load data-set
# http://yann.lecun.com/exdb/mnist/
mnist = tf.keras.datasets.mnist

# Convert the samples from integers to floating-point numbers
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras model by stacking layers and also optimizer, loss function are used for training:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and evaluate model:
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
