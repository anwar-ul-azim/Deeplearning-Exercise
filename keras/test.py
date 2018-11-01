# tensorflow installation check
import tensorflow as tf
from tensorflow import keras
import keras as k
print("version")
print(tf.__version__)
print(keras.__version__)
print(k.__version__)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

print(sess.run(hello))