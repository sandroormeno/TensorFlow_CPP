import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 32], name="x")
    y = tf.placeholder(tf.float32, [None, 8], name="y")

    w1 = tf.Variable(tf.truncated_normal([32, 16], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.0, shape=[16]))

    w2 = tf.Variable(tf.truncated_normal([16, 8], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.0, shape=[8]))

    a = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, w1), b1))
    y_out = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a, w2), b2), name="y_out")
    cost = tf.reduce_sum(tf.square(y-y_out), name="cost")
    optimizer = tf.train.AdamOptimizer().minimize(cost, name="train")
	
	# calcular el aprendizaje 
    prediccion_correcta = tf.equal(tf.argmax(y_out), tf.argmax(y) , name="prediccion")
    aprendizaje = tf.reduce_sum(tf.cast(prediccion_correcta, tf.float32) ,  name="aprendizaje")

    init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
    tf.train.write_graph(sess.graph_def, './', 'sandro.pb', as_text=False)