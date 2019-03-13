import tensorflow as tf

#Learning Data ( X, Y )
x_train = [1,2,3,4]
y_train = [6,5,7,10]

#declear Variable
W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

#define Hypothesis ( y = Wx + b )
hypothesis = x_train*W + b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#Minimize, optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph
sess.run(tf.global_variables_initializer()) # 변수형 자료형을 사용하기 전에 초기화 하는 작업

#training and find a optimum value
for step in range(2001):
    sess.run(train)
    if step % 100 == 0 :
        print(step,sess.run(cost), sess.run(W), sess.run(b))

'''
import tensorflow as tf

x = tf.constant([[1., 3.], [2., 6.]])

sess = tf.Session()

print(sess.run(x))
print(sess.run(tf.reduce_mean(x))) # total reduce_mean
print(sess.run(tf.reduce_mean(x, 0))) # column reduce_mean 열 단위
print(sess.run(tf.reduce_mean(x, 1))) # row reduce_mean 행 단위 

sess.close()

'''