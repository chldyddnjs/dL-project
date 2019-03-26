import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",onehot=True)
nb_casses = 10

X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32,[None,nb_classes])

W = tf.Variable(tf.random_normal([None,784]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost =  tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.eqaul(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(cast(is_correct,tf.float32))
num_epochs = 15
batch_size =100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        avg_cost = 0

        for i in range(num_iteration):
            batch_xs, batch_ys =mnist.train.next_batch(batch_size)
            _,cost_val = sess.run([train,cost],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += cost_val/ num_iteration

        print("Epoch:{:04d}, cost:{:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    print("Accuracy",
          accuracy.eval(session=sess,feed_dict={X:mnist.test.images,Y:mnist.test.labels}
                        ),
          )
    r = random.randint(0,mnist.test.num_examples -1)
    print("Label: ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("prediction:",sess.run(tf.argmax(mnis.hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys',interpolation='nearest')
    plt.show()
