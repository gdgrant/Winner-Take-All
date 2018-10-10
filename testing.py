import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

lid = tf.placeholder(tf.int32, [], 'lid')
gate = tf.get_variable('lesion', initializer=np.float32(np.ones([1,20])), trainable=False)
make = tf.assign(gate[:,lid], tf.zeros_like(gate)[:,lid])

with tf.Session() as sess:
    for i in range(20):
        print('Lesioning neuron {}:'.format(i))
        sess.run(tf.global_variables_initializer())
        print(np.int8(sess.run(gate)))
        sess.run(make, feed_dict={lid:i})
        print(np.int8(sess.run(gate)))
