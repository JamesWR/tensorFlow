import tensorflow as tf

import numpy
def demonstrateLinearEstimation():
	oneVal1 = tf.constant(3.5, dtype=tf.float32)
	oneVal2 = tf.constant(100.0)
	addNode = tf.add(oneVal1, oneVal2)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	inputAdd = tf.add(x, y)

	changeable1 = tf.Variable([.3], dtype=tf.float32)
	changeable2 = tf.Variable([.5], dtype=tf.float32)

	linear_model = changeable1 * x + changeable2

	error = tf.reduce_sum(tf.square(linear_model - y))

	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(error)

	x_train = [1,2,3,4]
	y_train = [0,-1,-2,-3]

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) # reset valuse to wrong
	for i in range(1000):
		sess.run(train, {x:x_train, y:y_train})
		trainingSet = {x:x_train, y:y_train}
		runList = [changeable1, changeable2, error]
		curr_1, curr_2, curr_error = sess.run(runList,trainingSet) 
		print("i: %d 1: %s 2: %s error:%s"%(i,curr_1,curr_2,curr_error))

	curr_1, curr_2, curr_error = sess.run([changeable1, changeable2, error], {x:x_train, y:y_train}) 
	print("i: %d 1: %s 2: %s error:%s"%(i,curr_1,curr_2,curr_error))

	#print(sess.run(inputAdd, {x:3,y:5}))
	#print(sess.run(inputAdd, {x:[1,2],y:[3,2]}))
demonstrateLinearEstimation()
