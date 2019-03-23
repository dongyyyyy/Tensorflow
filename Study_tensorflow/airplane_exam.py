import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#데이터셋을 섞음(train/test data set)
def shuffle_data(x_train,y_train): # 학습을 반복할 때마다(epoch) 데이터의 입력순서를 바꾸기 위함
    temp_index = np.arange(len(x_train))

    #Random suffle index
    np.random.shuffle(temp_index)

    #Re-arrange x and y data with random shuffle index
    x_temp = np.zeros(x_train.shape)
    y_temp = np.zeros(y_train.shape)
    x_temp = x_train[temp_index]
    y_temp = y_train[temp_index]

    return x_temp, y_temp

# 0에서 1 사이 값으로 정규화
#입력한 데이터셋에 최소값과 최대값을 찾은 후 전체 데이터를 0과 1 사이로 치환
def minmax_normaliztion(x):
    xmax , xmin = x.max() , x.min()
    return (x-xmin) / (xmax - xmin)

#정규화 후 realx에 해당하는 정규값 리턴
#변환하고 싶은 값과 전체 데이터셋을 입력하면 정규화된 범위 내에서 치환된 값을 리턴
def minmax_get_norm(realx, arrx):
    xmax, xmin = arrx.max() , arrx.min()
    normx = (realx - xmin) / (xmax - xmin)
    return normx

#0에서 1 사이 값을 실제 값으로 역정규값 리턴
#norm과 반대로 기존 값에서 0~1사이로바꾸는 것을 거꾸로 0~1사이의 값을 기존의 값으로 변경 그렇기에 역정규값 리턴이라 함.
def minmax_get_denorm(normx,arrx):
    xmax, xmin = arrx.max() , arrx.min()
    realx = normx * ( xmax - xmin ) + xmin
    return realx

def main():
    traincsvdata = np.loadtxt('./airplane_data/trainset.csv',unpack=True,delimiter=',',skiprows=1)
    num_point = len(traincsvdata[0])
    #print(traincsvdata)
    print('points : ' , num_point)

    # Speed(km/h) : [ 310. 312. 314. 319.
    x1_data =  traincsvdata[0]
    # Weight(ton) : [ 261. 2212. 261. 274. 221....
    x2_data = traincsvdata[1]
    # Distance(m) : [ 1814. 1632. 1981. 2163. 1733.
    y_data = traincsvdata[2]

    #빨간색(m)에 둥근점(o)로 시각화
    plt.plot(x1_data,y_data, 'mo')
    plt.suptitle('Traing set(x1)',fontsize=16)
    plt.xlabel('speed to take off')
    plt.ylabel('distance')
    plt.show()

    #파란색(b)에 둥금점(o)로 시각화
    plt.plot(x2_data , y_data , 'bo')
    plt.suptitle('Traing set(x2)', fontsize=16)
    plt.xlabel('weight')
    plt.ylabel('distance')
    plt.show()

    #데이터 정규화를 수행. 0~1 사이 값으로 변환
    x1_data = minmax_normaliztion(x1_data)
    x2_data = minmax_normaliztion(x2_data)
    y_data = minmax_normaliztion(y_data)

    #x_data 생성
    x_data = [[item for item in x1_data],
              [item for item in x2_data]]
    x_data = np.reshape(x_data,(-1,2))
    y_data = np.reshape(y_data,[len(y_data),1])

    #배치 수행
    BATCH_SIZE = 5
    BATCH_NUM = int(len(x1_data)/BATCH_SIZE)

    #총 개수는 정해지지 않았고 1개씩 들어가는 Placeholder 생성
    input_data = tf.placeholder(tf.float32 , shape=[None,2])
    output_data = tf.placeholder(tf.float32, shape=[None,1])

    #레이어 간 Weight 정의 후 랜덤값으로 초기화. 그림에서는 선으로 표시.
    W1 = tf.Variable(tf.random_uniform([2,5],0.0,1.0))
    W2 = tf.Variable(tf.random_uniform([5,3],0.0,1.0))
    W_out = tf.Variable(tf.random_uniform([3,1],0.0,1.0))

    #레이어의 노드가 하는 계산. 이전 노드와 현재 노드의 곱셈
    #비선형함수로 sigmoid추가
    hidden1 = tf.nn.sigmoid(tf.matmul(input_data,W1))
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1,W2))
    output = tf.matmul(hidden2,W_out)

    #비용함수, 최적화 함수, train 정의
    loss = tf.reduce_mean(tf.square(output-output_data))
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    #변수 사용 준비
    init = tf.global_variables_initializer()
    #세션 열고 init 실행
    sess =tf.Session()
    sess.run(init)

    #학습을 반복하며 값 업데이트
    for step in range(1000):
        index = 0
        #매번 데이터셋을 섞음
        x_data , y_data = shuffle_data(x_data,y_data)

        #배치크기 만큼 학습을 진행
        for batch_iter in range(BATCH_NUM-1):
            feed_dict = {input_data : x_data[index:index+BATCH_SIZE],
                         output_data : y_data[index:index+BATCH_SIZE]}
            sess.run(train,feed_dict=feed_dict)
            index+=BATCH_SIZE

    print('학습 완료. 임의의값으로 이륙거리 추정')
    arr_ask_x = [[290,210],[320,210],[300,300],[320,300]]

    for i in range(len(arr_ask_x)):
        ask_x = [arr_ask_x[i]]
        ask_norm_x = [[minmax_get_norm(ask_x[0][0], traincsvdata[0]),minmax_get_norm(ask_x[0][1],traincsvdata[1])]]

        answer_norm_y = sess.run(output,feed_dict = {input_data : ask_norm_x})
        answer_y = minmax_get_denorm(answer_norm_y,traincsvdata[2])

        print('이륙거리계산) 이륙속도(x1) : {}km/h, 비행기무게(x2) : {}ton, 이륙거리(y) : {}m'.format(ask_x[0][0],ask_x[0][1],answer_y[0][0]))


    #데스트셋 파일 읽음
    test_csv_x_data = np.load('./airplane_data/testset_x.csv',unpack=True,delimiter=',',skiprows=1)
    test_csv_y_data = np.load('./airplane_data/testset_y.csv',unpack=True,delimiter=',',skiprows=1)

    #속도와 무게의 테스트 값을 변수에 저장
    test_x1_data = test_csv_x_data[0]
    test_x2_data = test_csv_x_data[1]

    #테스트셋 정규화 진행
    test_x1_data = minmax_normaliztion(test_x1_data)
    test_x2_data = minmax_normaliztion(test_x2_data)
    test_csv_y_data = minmax_normaliztion(test_csv_y_data)

    #testset의 x_data 생성
    test_x_data = [[item for item in test_x1_data],
                   [item for item in test_x2_data]]
    test_x_data = np.reshape(test_x_data,len(test_x1_data)*2,order='F')
    test_x_data = np.reshape(test_x_data, (-1,2))

    #테스트셋 cvs파일 : 빨간색(m)에 둥근점(o)로 시각화
    #필자가 임의로 내림차순으로 정렬해 놓음
    plt.plot(list(range(len(test_csv_y_data))), test_csv_y_data,'mo')

    #예측데이터 : 검은색(k) 별표(*)로 시각화
    feed_dict = {input_data : test_x_data}
    test_pred_y_data = minmax_get_denorm(sess.run(output,feed_dict=feed_dict),traincsvdata[2])
    plt.plot(list(range(len(test_csv_y_data))),test_pred_y_data,'k*')

    #그래프 표시
    plt.suptitle('Test Result' , fontsize=16)
    plt.xlabel('index(x1,x2)')
    plt.ylabel('distance')
    plt.show()

main()