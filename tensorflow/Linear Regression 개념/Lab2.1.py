import tensorflow as tf
w= tf.Variable(tf.random_normal([1]), name='weight')
b= tf.Variable(tf.random_normal([1]), name='bias')
x=tf.placeholder(tf.float32, shape=[None])
y=tf.placeholder(tf.float32, shape=[None])
hypothesis = x*w+b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
sess =tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train],
        feed_dict={x: [1, 2, 3, 4, 5], y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step%20==0:
        print(step, cost_val, w_val, b_val)
"""
placeholder를 사용하는 이유는 만들어진 모델에 입력을 할수 있다는게 장점
"""
