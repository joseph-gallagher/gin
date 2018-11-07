import tensorflow as tf
import numpy as np

"""
A test to make sure that:

1) We can trigger updates only at the end of a sequence of inputs.
2) We can effectively compute the running lambda-discounted gradients.

"""

l = 0.7
x_data = np.random.randn(2000, 2)
w_real = np.array([[2, 0], [0, 3]])
noise = np.random.randn(2000, 2)*0.05

y_data = np.matmul(x_data, w_real) + noise

BATCH_SIZE = 2

g = tf.Graph()
with g.as_default():

    x = tf.placeholder(tf.float32, [2])
    y = tf.placeholder(tf.float32, [2])

    w = tf.Variable([[0, 0], [0, 0]], dtype=tf.float32)
    x_exp = tf.expand_dims(x, 0)
    y_pred_exp = tf.matmul(x_exp, w)
    y_pred = y_pred_exp[0]

    loss = tf.reduce_mean(tf.square(y_pred-y))

    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss)

    # Let's see what a grad of a matrix looks like.
    sum_grads = tf.Variable([[0, 0], [0, 0]], dtype=tf.float32)

    update_sum_grads = sum_grads.assign(
        grads[0][0] + l*sum_grads[0])
    const_grad = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    # simple_update = optimizer.apply_gradients([(const_grad, w)])
    update_final = optimizer.apply_gradients(
        [(sum_grads, w)])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(BATCH_SIZE):
            print(sess.run([grads, sum_grads, update_sum_grads],
                           {x: x_data[i], y: y_data[i]}))
        sess.run(update_final)
        hmm = sess.run(w)
        print(hmm)


class Foo:

    def __init__(self, a):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.const = tf.Variable(a, dtype=tf.float32)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

    def evaluate(self):

        with self.graph.as_default():
            evaluated = self.sess.run(self.const)
        return evaluated


# Learning how to expand/contract tensors in TensorFlow.

g = tf.Graph()
sess = tf.Session(graph=g)
with g.as_default():
    a = tf.Variable(0, tf.int32)
    b = tf.Variable(3, tf.int32)
    change_a = a.assign(7)
    change_b = b.assign(7)
    init = tf.global_variables_initializer()
sess.run(init)
sess.run([change_a, change_b])
print(sess.run([a, b]))
sess.run(a.initializer)
print(sess.run([a, b]))

# What is the result of




