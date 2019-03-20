import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#데이터셋을 섞음(train/validation/test data set)
def shuffle_data(x_train,y_train):
    temp_index = np.arange(len(x_train))

    #Random suffle index
    np.random.shuffle(temp_index)

    #Re-arrange x and y data with random shuffle index
    x_temp = np.zeros(x_train.shape)
    y_temp = np.zeros(y_train.shape)
    x_temp = x_train[temp_index]
    y_temp = y_train[temp_index]

    return x_temp, y_temp

def main():
    num_points = 5000 # (X,Y)데이터 개수
    vectors_set = []
    for i in range(num_points):
        x1=np.random.normal(.0,1.0)
        y1=np.sin(x1) + np.random.normal(.0,0.1)
        vectors_set.append([x1,y1])

    x_data = [v[0]for v in vectors_set]
    y_data = [v[1]for v in vectors_set]

    plt.plot(x_data,y_data,'go')
    plt.legend()
    plt.show()

    #배치 수행단위
    BATCH_SIZE = 100
    BATCH_NUM = int(len(x_data)/BATCH_SIZE)

    #데이터를 세로로(한 개씩) 나열한 형태로 reshape
    x_data = np.reshape(x_data , [len(x_data),1])
    y_data = np.reshape(y_data , [len(y_data),1])

    # 총 개수는 정해지지 않았고 1개씩 들어가는 Placeholder 생성
    input_data = tf.placeholder(tf.float32,shape=[None,1])
    output_data = tf.placeholder(tf.float32,shape=[None,1])


    #레이어 간 Weight 정의 후 랜덤값으로 초기화
    W1 = tf.Variable(tf.random_uniform([1,5],-1.0,1.0))
    W2 = tf.Variable(tf.random_uniform([5,3],-1.0,1.0))
    W_out = tf.Variable(tf.random_uniform([3,1],-1.0,1.0))

    #레이어의 노드가 하는 계산. 이건 노드의 현재 노드의 곱셈
    #비선형함수로 sigmoid 추가.
    hidden1 = tf.nn.sigmoid(tf.matmul(input_data,W1))
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2))
    output = tf.matmul(hidden2,W_out)

    #비용함수, 최적화함수, train정의
    loss = tf.reduce_mean(tf.square(output-output_data))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    #변수(Variable) 사용 준비
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(5001):
        index = 0
        x_data, y_data = shuffle_data(x_data,y_data)

        for batch_iter in range(BATCH_NUM-1):
            #origin
            feed_dict = {input_data: x_data[index:index+BATCH_SIZE],output_data:y_data[index:index+BATCH_SIZE]}
            sess.run(train,feed_dict= feed_dict)
            index += BATCH_SIZE
        if (step % 100 == 0) or (step <100 and step%10==0):
            print('Step=%5d, Loss Value=%f' % (step, sess.run(loss,feed_dict=feed_dict)))

    #5000번의 학습이 끝난 후 그래프로 결과 확인
    feed_dict = {input_data:x_data}
    #학습 데이터는 green 색(g)의 둥근점(o)으로 시각화
    plt.plot(x_data,y_data,'go')
    #예측 모델 출력은 검은색(k) 별로(*) 시각화
    plt.plot(x_data, sess.run(output,feed_dict=feed_dict),'k*')
    plt.xlabel('x')
    plt.xlim(-4,3)
    plt.ylabel('y')
    plt.ylim(-1.5,1.5)
    plt.legend()
    plt.show()

main()