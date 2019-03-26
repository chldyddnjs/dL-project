import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)} #char -> index 바꿔준다
 
dic_size = len(char2idx) # 입력의 갯수(one hot size)
hidden_size = len(char2idx) #출력의 갯수
num_classes = len(char2idx)#final output size
batch_size = 1 # one sample data, one batch
sequence_length = len(sample) - 1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # char to index 바꿔준다
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.int32, [None, sequence_length]) #X data
Y = tf.placeholder(tf.int32, [None, sequence_length]) #Y label

x_one_hot = tf.one.hot(X, num_classes) # one hot encoding
cell = tf.contrib.rnn.basicLSTMCell(
    num_units=hidden_size, state_is_tuple= True)
initial_state = cell.zero_state(batch_size, tf.float32)
output, _states = tf.nndynamic_rnn(
    cell, x_one_hot, initital_state=initial_state, dtype=tf.float32)
X_for_fc = tf.reshape(ouputs, [-1, hidden_size])# 퓰리커넥티드 레이어
outputs = tf.contrib.layers.fully_connected(X_for_fc,num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights)
loss = tf.reduce_mean(seqence_loss)
train = tf.AdamOptimizer(learning_rate=learning_rate)

prediction = tf.argmax(output, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l,_ = sess.run([loss, train], feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})

        result_str = [idx2char[c] for in np.squeeze(result)]

        print(i, "loss", l,"prediction", ''.join(result_str))
        
