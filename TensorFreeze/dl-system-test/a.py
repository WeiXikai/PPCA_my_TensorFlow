import tensorflow as tf
import numpy as np

a = tf.Variable([[1, 1], [1, 1], [1, 1], [1, 1]])
axis = [1, 1]
print("\n\n\n\n")
print(axis)
exit(0)
b = tf.reduce_sum(a, axis=axis)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(b))
