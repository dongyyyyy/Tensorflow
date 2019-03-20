#다수의 X값이 존재할 경우
import tensorflow as tf

# 데이터의 타입, 사이즈 정의 None = 무한대를 의미함
X1 = tf.placeholder(tf.float32,shape=[None])
X2 = tf.placeholder(tf.float32,shape=[None])

Y = tf.placeholder(tf.float32,shape=[None])

#변수 선언
W1 = tf.Variable(tf.random_normal([1]),name='weight1')
W2 = tf.Variable(tf.random_normal([1]),name='weight2')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

#가설
hypothesis = W1*X1 + W2*X2 + b

#cost function (예측값-실제결과값)^2
cost = tf.reduce_mean(tf.square(hypothesis-Y))

#learning rate & minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training
for step in range(5001):
    cost_val , W1_val, W2_val, b_val, _ = sess.run([cost,W1,W2,b,train],feed_dict={X1:[5,7] ,X2:[5,7],Y:[101,141]})
    if step % 500 == 0:
        print(step, cost_val, W1_val, W2_val,b_val)

# 예측값 출력
print(sess.run(hypothesis,feed_dict={X1:[8],X2:[8]}))

