import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None,784])
X_img = tf.reshape=[-1,28,28,1]
Y = tf.placeholder(tf.float32, [None,10])

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), )
#conv (?,28,28,32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)   #filter의 사이즈를 정해줌
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2 ,1],strides=[1,2,2,1], padding='SAME')

#maxpool shape = (?,14,14,32) dtype=float32
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.1))

L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],
                    strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#maxpool shape=(?,7,7,64)
W3 = tf.Variable(tf.random_normal([3,3,64,128]))

L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], stride=[1,2,2,1], padding='SAME')

L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1,128 *4*4])
#maxpool shape=(?,4,4,128)

W4 = tf.get_variable("W4", shape=[128*4*4,625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat,W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[625,10],
                     initializer=tf.contrib.layer.xaiver_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4,W5) +b5

cost = tf.reduuce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer = tf.train.AdamOprimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost =0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(toltal_batch):
        batch_xs ,batch_ys = mnist.train.next_batch(batch_size)
        feed_dict={X:batch_Ys,Y:batch_ys}
        c,_, = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += c/total_batch
    print("Epoch:", '%04d' (epoch +1 ), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finish')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.image, keep_prob:1}))

r = random.randint(0, mnist.test.num_examples -1)
print('Label:',sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print('prediction:',sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1], keep_prob:1}))





