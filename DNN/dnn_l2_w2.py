import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name = 'weight')
    b1 = tf.Variable(tf.random_normal([2], name = 'bias'))
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]))
    b2 = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
cost_hist = tf.summary.scalar('cost', cost) #for tensorboard
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
summary = tf.summary.merge_all() #for tensorboard

sess = tf.Session()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./compare/layer2-wide2')
writer.add_graph(sess.graph)
global_step = 0

for step in range(5000):
    s, _ = sess.run([summary, train], feed_dict = { X : x_data, Y : y_data })
    writer.add_summary(s, global_step=global_step)
    global_step += 1
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict = { X : x_data, Y : y_data }))

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = { X: x_data, Y : y_data })
print('\nHypothesis: ', h, '\nCorrect : ', c, '\nAccuracy : ', a)