#Placeholder
import tensorflow as tf
import numpy as np

#Build a graph.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.multiply(a,b) # 각 value의 곱
print('node c = ' , c)

#Launch the graph in a session
sess = tf.Session()

#Evaluate the tensor 'c'.
print('run c = ' , sess.run(c, feed_dict={a:3., b:4.}))

x = tf.placeholder(tf.float32,shape=(1024,1024)) # 1024 X 1024 배열로 선언
y = tf.matmul(x, x) # matmul = 행렬의 곱곱
rand_array = np.random.rand(1024,1024)
print(sess.run(y, feed_dict={x: rand_array}))