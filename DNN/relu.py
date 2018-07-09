import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


with tf.name_scope('layer1') as scope: #input layer
    W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0), name = 'weight1')
    b1 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias'))
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

with tf.name_scope('layer2') as scope: #hidden layer
    W2 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight2')
    b2 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias2'))
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

with tf.name_scope('layer3') as scope:
    W3 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight3')
    b3 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias3'))
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

with tf.name_scope('layer4') as scope:
    W4 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight4')
    b4 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias4'))
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)    

with tf.name_scope('layer5') as scope:
    W5 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight5')
    b5 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias5'))
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)    

with tf.name_scope('layer6') as scope:
    W6 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight6')
    b6 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias6'))
    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)    

with tf.name_scope('layer7') as scope:
    W7 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight7')
    b7 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias7'))
    layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)    

with tf.name_scope('layer8') as scope:
    W8 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight8')
    b8 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias8'))
    layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)    

with tf.name_scope('layer9') as scope:
    W9 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name = 'weight9')
    b9 = tf.Variable(tf.random_uniform([10], -1.0, 1.0, name = 'bias9'))
    layer9 = tf.nn.relu(tf.matmul(layer8, W9) + b9)    

with tf.name_scope('layer10') as scope: #output layer
    W10 = tf.Variable(tf.random_normal([10, 1], -1.0, 1.0), name = 'weight10')
    b10 = tf.Variable(tf.random_normal([1], -1.0, 1.0, name = 'bias10'))
    hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10)    

cost = -tf.reduce_mean( Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
cost_hist = tf.summary.scalar('cost', cost) #for tensorboard
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
summary = tf.summary.merge_all() #for tensorboard

sess = tf.Session()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./compare/relu')
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