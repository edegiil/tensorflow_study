import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

X = tf.placeholder(tf.float32, shape = [None, 784])
Y = tf.placeholder(tf.float32, shape = [None, 10])

keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('layer1') as scope: #input layer
    W1 = tf.get_variable('weight1', shape = [784, 200], initializer= tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_uniform([200], -1.0, 1.0, name = 'bias'))
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob = keep_prob)

with tf.name_scope('layer2') as scope: #hidden layer
    W2 = tf.get_variable('weight2', shape = [200, 200], initializer= tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_uniform([200], -1.0, 1.0, name = 'bias2'))
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob = keep_prob)

with tf.name_scope('layer10') as scope: #output layer
    W10 = tf.get_variable('weight10', shape = [200, 10], initializer= tf.contrib.layers.xavier_initializer())
    b10 = tf.Variable(tf.random_normal([10], -1.0, 1.0, name = 'bias10'))
    hypothesis = tf.sigmoid(tf.matmul(layer2, W10) + b10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis, labels = Y))
cost_hist = tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost) #AdamOptimzer is better for here
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epoch = 20
batch_size = 100

summary = tf.summary.merge_all()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./compare/dnn_mnist_dropout')
writer.add_graph(sess.graph)
global_step = 0

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for batch in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        s, c, _ = sess.run([summary, cost, optimizer], feed_dict = { X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        avg_cost += c / total_batch

        writer.add_summary(s, global_step=global_step)
        global_step += 1
        
    print('epoch:' ,epoch + 1, 'average cost : ', avg_cost)

print("Accuracy : ", accuracy.eval(session = sess, feed_dict = {X: mnist.test.images, Y:mnist.test.labels, keep_prob: 1.0}))

r = random.randint(0, 1000)
print("Labels : ", sess.run(tf.argmax(mnist.test.labels[r: r+1], 1)))
print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {X: mnist.test.images[r: r+1], keep_prob: 1.0}))
plt.imshow(mnist.test.images[r: r+1].reshape(28,28), cmap = 'Greys', interpolation='nearest')
plt.show()