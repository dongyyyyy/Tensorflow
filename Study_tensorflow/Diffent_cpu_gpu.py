import tensorflow as tf
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100,100,100,3))
    net_cpu = tf.layers.conv2d(random_image_cpu,32,7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100,100,100,3))
    net_gpu = tf.layers.conv2d(random_image_gpu,32,7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config = config)

try:
    sess.run(tf.global_variables_initializer())
except:
    print('\n\nThis error most likely means that this notebook is not'
          'configured use a GPU. Change this in Notebook Settings via the'
          'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
    raise

def cpu():
    sess.run(net_cpu)
def gpu():
    sess.run(net_gpu)


print('Time(s) to convolve 32 * 7 * 7 * 3 filter over random 100 * 100* 100 * 3 images'
      '(batch * height * width * channel). Sum of ten runs.')
print('CPU로 계산시간 (초) : ')
cpu_time = timeit.timeit('cpu()', number = 10 , setup = "from __main__ import cpu")
print(cpu_time)
print('GPU로 계산시간 (초) : ')
gpu_time = timeit.timeit('gpu()',number = 10 , setup="from __main__ import gpu")
print(gpu_time)
print('GPU의 속도는 CPU보다 {}x 배 빠릅니다.'.format(int(cpu_time/gpu_time)))

sess.close()