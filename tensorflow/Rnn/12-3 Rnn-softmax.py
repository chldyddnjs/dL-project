import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c : i for i,c in enumerate(idx2char)}

dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
seqence_length = len(sample) -1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # char to index
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

X = tf.placeholder(tf.float32,[None, sequence_length])
Y = tf.placeholder(tf.float32,[None, sequence_length])

X_one_hot = tf.one_hot(X,num_classes)
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])

softmax_w = tf.get_variable("softmax_w",[rnn_hidden_size,num_classes])
softmax_b = tf.get_variable("softmax_b",[num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l,_ = sess.run([loss, train], feed_dict={X:x_data,Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss",l,"prediction",''.join(result_str))
        
    
