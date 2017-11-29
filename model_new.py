import tensorflow as tf
import pandas as pd
import numpy as np
import image_gen as ig
import sqlite3
import random
import time
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

start_time = time.time()
print("Starting...")
r_seed = int(start_time)


# function to initialize weights with noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# function to initialize with positive biases (for ReLU) to avoid "dead neurones"
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution with output and input the same size
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max pooling over 2x2 blocks
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


print("Reading files...")
# import data
kanji_data = pd.read_table("joyojinmeiyo.txt", encoding='utf-16')

filename_suffix = '.tiff'
strat_col = kanji_data['LEX']

print("Creating arrays...")
kanji_WORD_array = np.zeros((len(kanji_data), 1024))
kanji_LEX_onehot = np.zeros((len(kanji_data), 24))

print("Loading images...")
# creates arrays of the words
for i in range(0, len(kanji_data)):
    kanji_WORD_array[i] = ig.load_image('images/' + kanji_data['WORD'][i] + filename_suffix)

print("Loading labels...")
# creates arrays of the lexicality
lexicality = list(kanji_data['LEX'].unique())
for i in range(0, len(kanji_data)):
    kanji_LEX_onehot[i] = np.eye(len(lexicality))[lexicality.index(kanji_data['LEX'][i])]

print("Splitting into training and testing sets...")
# split the data into testing and training sets
# TODO considering combining these before the split, but the shared parameters should suffice for now
training_images, testing_images = train_test_split(kanji_WORD_array, test_size=0.2, stratify=strat_col, random_state=r_seed)
training_labels, testing_labels = train_test_split(kanji_LEX_onehot, test_size=0.2, stratify=strat_col, random_state=r_seed)
training_data, testing_data = train_test_split(kanji_data, test_size=0.2, stratify=strat_col, random_state=r_seed)
training_data = training_data.reset_index(drop=True)
testing_data = testing_data.reset_index(drop=True)

# create a class for images and labels
class cnnData(object):
    def __init__(self, kanji, strokes, images, labels):
        self.kanji = kanji
        self.strokes = strokes
        self.images = images
        self.labels = labels

print("Instancing classes...")
# create instances of that class for training and testing
trainingData = cnnData(training_data["WORD"], training_data["LEX"], training_images, training_labels)
testingData = cnnData(testing_data["WORD"], testing_data["LEX"], testing_images, testing_labels)

# start an interactive session in TF
sess = tf.InteractiveSession()

# create nodes for the input images
x = tf.placeholder("float", shape=[None, 1024])  # None = first dimension; 64*64 = dimensionality of one image
y_ = tf.placeholder("float", shape=[None, 24])  # number of strokes possible in this dataset

# reshape x to a 4D tensor
x_image = tf.reshape(x, [-1, 32, 32, 1])  # 1- ???, 50x200 image, 1 color channel

# FIRST LAYER ::: takes 1 image and results in 32 32x32 feature maps

# initialize weights and biases to compute 32 features for each 8x8 patch
W_conv1 = weight_variable([3, 3, 1, 64])  # 8x8 patch, 1 input channel, 32 output channels
b_conv1 = bias_variable([64])  # for each of the 32 output channels

# implement the layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # convolve with weight tensor and add bias; apply ReLU
h_pool1 = max_pool_2x2(h_conv1)  # max pooling

# SECOND LAYER ::: takes 32 feature maps and results in 64 16x16 feature maps

# initialize weights and biases to compute 64 features for each 6x6 patch
W_conv2 = weight_variable([3, 3, 64, 128])  # 6x6 patch, 32 input channels, 64 output channels
b_conv2 = bias_variable([128])  # for each of the 64 output channels

# implement the layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD LAYER ::: takes 64 feature maps and results in 128 8x8 feature maps

# initialize weights and biases to compute 128 features for each 4x4 patch
W_conv3 = weight_variable([3, 3, 128, 256])  # 4x4 patch, 64 input channels, 128 output channels
b_conv3 = bias_variable([256])  # for each of the 128 output channels

# implement the layer
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# DENSELY (FULLY) CONNECTED LAYER

# initialize weights and biases to process the entire image
W_fc1 = weight_variable([4 * 4 * 256, 500])  # 8x8x128 (8192) for the 128 8x8 feature maps, 500 nodes in this layer
b_fc1 = bias_variable([500])

# reshape the tensor into a batch of vectors
h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*256])  # -1 ???, 8x8x128 to be flattened (8192)

# implement the layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # multiply by weights and add bias; apply ReLU

# dropout (reduces overfitting)
keep_prob = tf.placeholder("float")  # probability of neuron output being kept during dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER

# initialize weights and biases to produce final output
W_fc2 = weight_variable([500, 24])  # input of 500 nodes, output to binary decision
b_fc2 = bias_variable([24])

# implement the layer
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # multiply by weights and add bias; apply SoftMax

# training and evaluation
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # cost function: cross entropy between target and prediction
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # 1e-5 learning rate
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # check accuracy; boolean result
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # convert boolean to float -> take the mean
sess.run(tf.global_variables_initializer())  # required


# set variables
summaries_dir = "summary/"
batch_size = 100
num_trials = 1000000
test_freq = 10000

# prepare the database
print("Preparing environment...")

# create summaries directory
if tf.gfile.Exists(summaries_dir):  # if the summaries directory exists, delete it and everything in it
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)  # create the summaries directory

# prepare the SQLite database
conn = sqlite3.connect(summaries_dir + 'results.db', timeout=10)  # create the results database
c = conn.cursor()  # create a cursor in the database
c.execute('''CREATE TABLE train (character TEXT, target INTEGER, prediction INTEGER, trials INTEGER)''')
c.execute('''CREATE TABLE test (character TEXT, target INTEGER, prediction INTEGER, trials INTEGER)''')
conn.commit()  # commit changes to the database
conn.close()  # disconnect from the database

print("Beginning training...")
# run through training
for i in range(int(num_trials / batch_size) + 1):

    batch_kanji, batch_strokes, batch_images, batch_labels = zip(*random.sample(list(zip(trainingData.kanji,
                                                                                         trainingData.strokes,
                                                                                         trainingData.images,
                                                                                         trainingData.labels)),
                                                                                batch_size))
    if i % int(test_freq / batch_size) == 0:  # testing frequency
        # print testing accuracy to console
        test_accuracy = accuracy.eval(feed_dict={x: testingData.images, y_: testingData.labels, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: trainingData.images, y_: trainingData.labels, keep_prob: 1.0})
        print("Elapsed time:", np.round(((time.time() - start_time) / 60), 2), "minutes,",
              "step %d, test acc %e, train acc %f " % (int(i * batch_size), test_accuracy, train_accuracy))

        # append TESTING results to database
        test_pred = y_conv.eval(feed_dict={x: testingData.images, y_: testingData.labels, keep_prob: 1.0})
        pred = np.argmax(test_pred, axis=1) # determine prediction

        # connect to database and create a cursor
        conn = sqlite3.connect(summaries_dir + 'results.db', timeout=10)
        c = conn.cursor()

        for k in range(len(testingData.kanji)):
            kanji = str(testingData.kanji[k])
            strokes = str(testingData.strokes[k])
            predict = str(pred[k] + 1)

            # write strain data to the database
            c.execute('''INSERT INTO test VALUES (?, ?, ?, ?)''', (kanji, strokes, predict, str(i * batch_size)))
            conn.commit()  # commit those changes to the database
        conn.close()

        # append TRAINING results to database
        train_pred = y_conv.eval(feed_dict={x: trainingData.images, y_: trainingData.labels, keep_prob: 1.0})
        pred = np.argmax(train_pred, axis=1)  # determine prediction

        # connect to database and create a cursor
        conn = sqlite3.connect(summaries_dir + 'results.db', timeout=10)
        c = conn.cursor()

        for k in range(len(trainingData.kanji)):
            kanji = str(trainingData.kanji[k])  # fix this
            strokes = str(trainingData.strokes[k])  # and this
            predict = str(pred[k] + 1)

            # write strain data to the database
            c.execute('''INSERT INTO train VALUES (?, ?, ?, ?)''', (kanji, strokes, predict, str(i * batch_size)))
            conn.commit()  # commit those changes to the database
        conn.close()

    # actually run training
    train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})

    # append ONLINE results to database
    # y_out = y_conv.eval(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0}) # evaluate on the testing set
    # pred = np.argmax(y_out, axis=1) # determine prediction
    # conn = sqlite3.connect(summaries_dir + 'results.db', timeout=10)
    # c = conn.cursor()
    # for k in range(batch_size):
    #     kanji = str(batch_kanji[k])  # fix this
    #     strokes = str(batch_strokes[k])  # and this
    #     predict = str(pred[k] + 1)
    #
    #     # write strain data to the database
    #     c.execute('''INSERT INTO train VALUES (?, ?, ?, ?)''', (kanji, strokes, predict, str(i * batch_size)))
    #     conn.commit()  # commit those changes to the database
    # conn.close()

# print the results using the test set
print("Final accuracy %g" % accuracy.eval(feed_dict={x: testingData.images, y_: testingData.labels, keep_prob: 1.0}))

# print the amount of time elapsed in minutes
elapsed_time = time.time() - start_time
print("Completed... Total time:", np.round((elapsed_time / 60), 2), "minutes.")

# TODO implement growing vocabulary methods
