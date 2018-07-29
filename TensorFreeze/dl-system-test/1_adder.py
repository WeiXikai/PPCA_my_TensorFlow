""" import your model here """
import TensorFreeze as tf

import numpy as np

sess = tf.Session()

# adder 
a = tf.placeholder(tf.float32, name = "a")
b = tf.placeholder(tf.float32, name = "b")
adder_node = a + b

ans = sess.run(adder_node, {a: 3, b: 4.5})
assert np.equal(ans, 7.5)

ans = sess.run(adder_node, {a: [1, 3], b: [2, 3]})
assert np.array_equal(ans, [3, 6])

print ("test: 1_adder finished")
