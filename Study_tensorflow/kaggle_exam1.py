import tensorflow as tf
import pandas
import numpy as np

# Load CSV file as matrix
train_csv_data = pandas.read_csv('./titanic_data/train.csv').as_matrix()
test_csv_data = pandas.read_csv('./titanic_data/test.csv').as_matrix()
test_csv_sub = pandas.read_csv('./titanic_data/gender_submission.csv').as_matrix()

# Male = 1 / Female = 0
for i in range(len(train_csv_data)):
    #print(train_csv_data[i,4]
    if train_csv_data[i,4] == 'male' : #남성일 경우
        train_csv_data[i,4] = 1
    else : # 여성일 경우
        train_csv_data[i,4] = 0

#test_data의 male,female변경
for i in range(len(test_csv_data)):
    if test_csv_data[i,3] == 'male':
        test_csv_data[i,3] = 1
    else :
        test_csv_data[i,3] = 0

#승선항에 대한 정보 또한 영문자이므로 수치화한다. 비어있을 경우 -> 0 S = 1 C = 2 Q = 3
for i in range(len(train_csv_data)):
    if train_csv_data[i,11] == 'S' :
        train_csv_data[i,11] = 1
    elif train_csv_data[i,11] == 'C':
        train_csv_data[i, 11] = 2
    elif train_csv_data[i,11] == 'Q' :
        train_csv_data[i, 11] = 3
    if np.isnan(train_csv_data[i,11]):
        train_csv_data[i, 11] = 0

for i in range(len(test_csv_data)):
    if test_csv_data[i,10] == 'S' :
        test_csv_data[i,10] = 1
    elif test_csv_data[i,10] == 'C':
        test_csv_data[i, 10] = 2
    elif test_csv_data[i,10] == 'Q' :
        test_csv_data[i, 10] = 3
    if np.isnan(test_csv_data[i,10]):
        test_csv_data[i, 10] = 0
## 전처리 완료 ##


X_PassengerData = train_csv_data[:,[2,4,6,7,11]] # 2 = Pclass , 4 = sex , 6 = SibSp , 7 = Parch , 11 = Embarked
Y_Survived = train_csv_data[:,1:2]
Test_X_PassengerData = test_csv_data[:,[1,3,5,6,10]]
Test_Y_Survived = test_csv_sub[:,1:2]

#placeholders
X = tf.placeholder(tf.float32, shape = [None,5])
Y = tf.placeholder(tf.float32, shape = [None,1])

W = tf.Variable(tf.random_normal([5,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

#hypothesis
hypothesis = tf.sigmoid(tf.matmul(X,W)+b) # 값이 크고 많기 때문에 matmul 행렬곱을 하여 계산한 후 sigmoid한다

#cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
# Y = 1 이면 생존 0은 사망. 따라서 cost함수에서 1과 0인 경우를 나누어 계산함.
# y = 1인경우 -> cost = -tf.reduce_mean(1*tf.log(hypothesis) + 0 이 된다.
# log(x)와 log(1-x)그래프 확인 ! log(x) x = 1이면 y = 0이고 x = 0 이면 y = inf이다.
#반대로 log(1-x)그래프에서는 x = 1이면 y = inf이고 x = 0이면 y = 0이다.

#Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

previous_cost = 0.

#Lanch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val,_ = sess.run([cost,train],feed_dict={X:X_PassengerData,Y:Y_Survived})

        if step%500 == 0:
            print('Step = ',step,", Cost = ", cost_val)

        if previous_cost == cost_val:
            print("found best hypothesis when step : ",step,"\n")

            break
        else :
            previous_cost = cost_val
    # Validation
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: X_PassengerData, Y: Y_Survived})
    print("\n Accuracy : ", a)

    print('\n Test CSV runningResult')

    h2, c2, a2 = sess.run([hypothesis, predicted, accuracy], feed_dict={X: Test_X_PassengerData, Y: Test_Y_Survived})
    print('\n Accaracy : ', a2)