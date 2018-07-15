from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2
import sys

file_model = './model/'
file_name="just_a_test_img.png"
loc_dst="./"
file_loc = loc_dst + file_name

class_nums = 6
class_bins = {0: 'shake_hands',
             1: 'hug',
             2: 'kick',
             3: 'point',
             4: 'punch',
             5: 'push'}

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    ## Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
    ## Activation.
    conv1 = tf.nn.relu(conv1)

    ## Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ## Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    ## Activation.
    conv2 = tf.nn.relu(conv2)

    ## Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ## Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    ## Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b

    ## Activation.
    fc1 = tf.nn.relu(fc1)

    ## Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    ## Activation.
    fc2 = tf.nn.relu(fc2)

    ## Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, class_nums), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(class_nums))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def imgPreprocessor(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Size of original images
    # max-min range --> (332-216, 612, 244)
    # avg range --> (260, 380)
    img_gray = cv2.resize(img_gray, (70, 95))
    img_gauss = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_norm = np.empty_like((img_gauss))
    img_norm = cv2.normalize(img_gauss, img_norm, 0, 255, cv2.NORM_MINMAX)
    return img_norm

def writeImg(img):
    cv2.imwrite(file_loc, img)

dir_ext = './data/5_9_48.png'
if len(sys.argv)>1:
    dir_ext = sys.argv[1]

try:
    class_name = class_bins[int(dir_ext.split('/')[2].split('_')[0])]
except:
    class_name = 3  # incase filename is bad!

def main():
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    logits = LeNet(x)
    saver = tf.train.Saver()

    img = imgPreprocessor(cv2.imread(dir_ext))
    writeImg(img)
    img = cv2.imread(file_loc)
    img = cv2.resize(img, (32, 32))
    img = img[:,:,0]
    img = ((img-255)/255)
    test_img = img
    test_img = test_img.reshape(1, 32, 32, 1)

    with tf.Session() as sess:
        # saver.restore(sess, file_model)
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        output = sess.run(logits, feed_dict={x: test_img})
        print(f"\nSoftmax --> \n {output}")
        print()
        print(f'Model prediction -> {class_bins[np.argmax(output)]}, ({np.argmax(output)})')

    plt.imshow(plt.imread(dir_ext))
    plt.title(f'Actual - {class_name} || Prediction - {class_bins[np.argmax(output)]}')
    plt.show()

if __name__ == '__main__':
    main()
