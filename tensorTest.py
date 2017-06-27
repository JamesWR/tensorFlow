import tensorflow as tf
oneVal1 = tf.constant(3.5, dtype=tf.float32)
oneVal2 = tf.constant(100.0)
addNode = tf.add(oneVal1, oneVal2)
placholder1 = tf.placeholder(tf.float32)
placholder2 = tf.placeholder(tf.float32)
inputAdd = tf.add(placholder1, placholder2)

changeable1 = tf.Variable([2], dtype=tf.float32)
changeable2 = tf.Variable([.5], dtype=tffloat32)



sess = tf.Session()
print(sess.run(inputAdd, {placholder1:3,placholder2:5}))
print(sess.run(inputAdd, {placholder1:[1,2],placholder2:[3,2]}))
