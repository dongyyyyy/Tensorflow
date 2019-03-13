from __future__ import division
from __future__ import print_function
import os.path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist_data',one_hot=True)

for i in np.random.randint(55000, size = 6):
    imgvec = mnist.train.images[i,:]
    labelvec = mnist.train.labels[i,:]
    imgmatrix = np.reshape(imgvec, (28,28)) # (784 ,) -> (28 ,28)
    label = np.argmax(labelvec) # [ 0 0 1 ... ] -> 2
    plt.matshow(imgmatrix, cmap=plt.get_cmap('gray'))
    plt.title('Index : %d, Label : %d' %(i,label))

