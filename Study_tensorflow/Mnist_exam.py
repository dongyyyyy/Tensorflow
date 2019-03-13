import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist_data/",one_hot=True)

# 변수들을 설정
# mnist 이미지데이터 형태는 28*28 = 784
X = tf.placeholder(tf.float32,[None,784])
# 결과 : 0 ~ 9
Y = tf.placeholder(tf.float32,[None,10])


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
logit_y = tf.matmul(X,W)+b

# sofmax & cross-entropy
softmax_y = tf.nn.softmax(logit_y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(softmax_y), reduction_indices=[1]))
train_step  =tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    #배치 크기는 100
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X:batch_xs,Y:batch_ys})

#확인
correct_prediction = tf.equal(tf.argmax(softmax_y,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('정확도 : ' , sess.run(accuracy,feed_dict={X:mnist.test.images, Y : mnist.test.labels}))