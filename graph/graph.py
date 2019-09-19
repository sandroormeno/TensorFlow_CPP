import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	a = tf.Variable(5.0, name='a')
	b = tf.Variable(6.0, name='b')
	i = tf.Variable([[1,2],[3,4]], dtype=tf.float32,   name="i") # se debe poner el tipo de variable
	u = tf.Variable([[10,5],[10,5]], dtype=tf.float32 ,  name="u")
	so = tf.add(i, u, name='so')	
	mul = tf.matmul(i, u, transpose_b=True, name='mul')
	s= tf.add(a, b, name='s')
	sess.run(tf.global_variables_initializer())
	#print(s.eval())
	print(mul.eval())
	tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb', as_text=False)