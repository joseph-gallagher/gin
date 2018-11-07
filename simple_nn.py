import tensorflow as tf
import numpy as np

"""
This code creates the TensorFlow code that generates the graph of a simple_nn player.
"""


class SimpleGraph(object):

    def __init__(self, path=None, hidden_units=30, alpha=0.1, gamma=0.7):

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.alpha = alpha
        self.gamma = gamma

        with self.graph.as_default():

            with tf.name_scope('compute') as scope:

                self.state = tf.placeholder(tf.float32, [None, 52])
                self.w1 = tf.Variable(1/52*np.random.randn(52, hidden_units), dtype=tf.float32)
                self.hidden = tf.sigmoid(tf.matmul(self.state, self.w1))
                self.w2 = tf.Variable(np.random.randn(hidden_units, 1), dtype=tf.float32)
                self.end_diff = tf.matmul(self.hidden, self.w2)

                # This should be updated every time we advance the game.
                self.last_end_diff = tf.Variable(0, dtype=tf.float32)
                self.update_last_end_diff = self.last_end_diff.assign(self.end_diff[0][0])

                # These only enter to feed in the final score.
                self.final_score = tf.Variable(0, dtype=tf.float32)
                self.feed_final_score = tf.placeholder(tf.float32)
                self.update_final_score = self.final_score.assign(self.feed_final_score)

            with tf.name_scope('grads') as scope:

                optimizer = tf.train.GradientDescentOptimizer(alpha)

                # Currently holding separate sums for both gradients.
                # This is a kludge; what is really needed is a class
                # extending GradientDescentOptimizer that can handle
                # updates for all variables simultaneously.

                self.w1_grads = optimizer.compute_gradients(self.end_diff[0][0], [self.w1])
                self.w1_sum_grads = tf.Variable(np.zeros((52, hidden_units)), dtype=tf.float32)
                self.w1_sum_grads_update = self.w1_sum_grads.assign(
                    self.w1_grads[0][0] + self.gamma*self.w1_sum_grads
                )

                self.w2_grads = optimizer.compute_gradients(self.end_diff[0][0], [self.w2])
                self.w2_sum_grads = tf.Variable(np.zeros((hidden_units, 1)), dtype=tf.float32)
                self.w2_sum_grads_update = self.w2_sum_grads.assign(
                    self.w2_grads[0][0] + gamma*self.w2_sum_grads
                )

                self.w1_change = tf.Variable(np.zeros((52, hidden_units)), dtype=tf.float32)
                self.w2_change = tf.Variable(np.zeros((hidden_units, 1)), dtype=tf.float32)

                self.w1_change_update = self.w1_change.assign(
                    (self.end_diff[0][0] - self.last_end_diff)*self.w1_sum_grads + self.w1_change
                )
                self.w2_change_update = self.w2_change.assign(
                    (self.end_diff[0][0] - self.last_end_diff)*self.w2_sum_grads + self.w2_change
                )

                self.w1_change_update_final = self.w1_change.assign(
                    (self.final_score - self.last_end_diff)*self.w1_sum_grads + self.w1_change
                )
                self.w2_change_update_final = self.w2_change.assign(
                    (self.final_score - self.last_end_diff)*self.w2_sum_grads + self.w2_change
                )

                # These are not the gradients to apply; need to use the TD-formula.
                self.update_all = optimizer.apply_gradients([(self.w1_change, self.w1),
                                                            (self.w2_change, self.w2)])

            with tf.name_scope('setup') as scope:
                init = tf.global_variables_initializer()

            self.sess.run(init)
            self.saver = tf.train.Saver()

        self.path = path
        if self.path:
            # We are assuming we are only keeping one checkpoint.
            recovery_path = self.path + '-0'
            self.saver.restore(self.sess, recovery_path)

    def end_game(self, end_score):

        # Use the special-built method to calculate the final delta w
        self.sess.run(self.update_final_score, {self.feed_final_score: end_score})
        self.sess.run([self.w1_change_update_final, self.w2_change_update_final])

        # Reset all the variables except the updates w1_change and w2_change.
        self.sess.run(self.last_end_diff.initializer)
        self.sess.run(self.final_score.initializer)
        self.sess.run(self.w1_sum_grads.initializer)
        self.sess.run(self.w2_sum_grads.initializer)

    def learn(self):

        # Update the weights and then reset w1_changes, w2_changes.
        self.sess.run(self.update_all)
        self.sess.run(self.w1_change.initializer)
        self.sess.run(self.w2_change.initializer)

    def save_state(self, save_path=None):

        if self.path:
            self.saver.save(self.sess, self.path, global_step=0)
        else:
            if save_path:
                self.saver.save(self.sess, save_path, global_step=0)
                self.path = save_path

    def evaluate(self, data, training=False, first_turn=False):

        # Takes in a one-hot representation of the deck and computes
        # an evaluated score for it.
        with self.graph.as_default():
            evaluation = self.sess.run(self.end_diff, {self.state: data})
            if training:
                # First play is different; there is no update to w{i}_change.
                if not first_turn:
                    self.sess.run([self.w1_change_update,
                                   self.w2_change_update], {self.state: data})

                self.sess.run([self.update_last_end_diff,
                               self.w1_sum_grads_update,
                               self.w2_sum_grads_update], {self.state: data})

        return evaluation








