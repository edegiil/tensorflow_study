import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([2, 10]), name = 'weight1')
    b1 = tf.Variable(tf.random_normal([10], name = 'bias'))
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([10, 10]), name = 'weight2')
    b2 = tf.Variable(tf.random_normal([10], name = 'bias2'))
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope('layer3') as scope:
    W3 = tf.Variable(tf.random_normal([10, 1]), name = 'weight3')
    b3 = tf.Variable(tf.random_normal([1], name = 'bias3'))
    hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)

cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
cost_hist = tf.summary.scalar('cost', cost) #for tensorboard
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
summary = tf.summary.merge_all() #for tensorboard

sess = tf.Session()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./compare/layer3-wide10')
writer.add_graph(sess.graph)
global_step = 0

for step in range(10001):
    s, _ = sess.run([summary, train], feed_dict = { X : x_data, Y : y_data })
    writer.add_summary(s, global_step=global_step)
    global_step += 1
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict = { X : x_data, Y : y_data }))

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = { X: x_data, Y : y_data })
print('\nHypothesis: ', h, '\nCorrect : ', c, '\nAccuracy : ', a)