import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)

if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denomiator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator + 1e-7)


seq_length = 7
data_dim =5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
ireations= 500

xy = np.loadtxt('https://github.com/hunkim/DeepLearningZeroToAll/blob/master/data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]

train_size =int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length]

train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i+swq_length, :]
        _y = time_series[i+seq_length, [-1]]

        print(_x,"->",_y)
        dataX.append(_x)
        dataY.append(-y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

cell = tf.contrib.rnn.BacsicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:-1], output_dim, activation_fn=None)

loss =tf.reduce_sum(tf.square(Y_pred -Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train =optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None,1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - prediction)))

with tf.Session() as sess:
    init = tf.globale_variables_initializer()
    sess.run(init)

    for i in range(iterations):
        _,step_loss = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

        test_predict = sess.run(Y_pred, feed_dict={X:testX})
        rmse_val = sess.run(rmse, feed_dict={
            targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))

        plt.plot(TestY)
        plt,plot(test_predict)
        plt.xlabel("time perid")
        plt.ylabel("stock price")
        plt.show()
