import tensorflow as tf
import matplotlib.pyplot as plt
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess =tf.Session()
sess.run(tf.global_variables_initializer())
W_val = []
Cost_val = []
for i in range(-30, 50):
    feed_W = i*0.1
    curr_cost, curr_W = sess.run([cost,W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    Cost_val.append(curr_cost)


plt.plot(W_val, Cost_val)
plt.show()
