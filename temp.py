import tensorflow as tf
import numpy as np

input1 = tf.keras.Input(shape=(3,))
input2 = tf.keras.Input(shape=(3,))
input3 = tf.keras.Input(shape=(1,))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(input1)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
outputs = input1 * input3 + input2
model = tf.keras.Model(inputs=[input1, input2, input3], outputs=outputs)

out = model([np.array([[1, 2, 3], [3, 2, 1]]), np.array([[3, 2, 1], [1, 2, 3]]), np.array([[100], [200]])])
print(out)
