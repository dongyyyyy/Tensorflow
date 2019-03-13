#국어 점수 예측하기
import tensorflow as tf

x_train = tf.placeholder(tf.float32,shape = [None])
y_train = tf.placeholder(tf.float32,shape = [None])

W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

hypothesis = x_train*W + b

cost = tf.reduce_mean(tf.square(hypothesis-y_train)) # tf.square = 제곱 연산


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val , W_val , b_val , _ = sess.run([cost,W,b,train],feed_dict={x_train:[5,7] , y_train:[52,72]})
    if step % 100 == 0:
        print(step,cost_val,W_val,b_val)

print(8 ,"예측 Y : " , sess.run(hypothesis,feed_dict={x_train:[8]}))
