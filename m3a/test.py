import numpy as np
import tensorflow as tf

# Create a TensorFlow tensor
x = tf.constant([[1, 2, 3, 4, 5], [4, 3, 2,1 ,5], [1 ,2, 3, 4, 5]])

# Create a NumPy array for indexing
indices = np.array([1, 0])

# Convert the NumPy array to a TensorFlow tensor
indices_tensor = tf.convert_to_tensor(indices)

# Use the indices tensor to index the original tensor
selected_values = tf.gather(x, indices_tensor)

print(selected_values)