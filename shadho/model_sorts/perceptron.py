"""
Online reinforcement learning perceptron using mini-batching
"""
import numpy as np
import tensorflow as tf

from IPython.core.debugger import Tracer

class Perceptron(object):
    """
    Online reinforcement learning perceptron for mapping models to compute
    classes
    """
    def __init__(self, input_length, target_levels, model_ids, compute_class_ids, output_levels=None, *args, **kwargs):
        self.model_ids = np.array(model_ids)
        self.compute_class_ids = np.array(compute_class_ids)

        self.network_input, self.softmax_linear = self.inference(input_length, target_levels, output_levels)
        self.total_loss, self.reinforcement_penalties= self.loss(self.softmax_linear, input_length)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = self.train(self.total_loss, self.global_step)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess = tf.Session()
        self.sess.run(init_op)

    def model_id_to_onehot(self, model_id):
        return (self.model_ids == model_id).astype(int)

    def compute_class_to_onehot(self, compute_class):
        return (self.compute_class_ids == compute_class).astype(int)

    def handle_input(self, raw_input_vectors):
        """Handles multiple raw input vectors."""
        input_vectors = []
        for i in raw_input_vectors:
            input_vectors.append(np.append(np.append(self.model_id_to_onehot(i[0]), self.compute_class_to_onehot(i[1])),  i[2:]).reshape([1, -1]))
        return input_vectors

    def handle_output(self, raw_output):
        """Handles single raw_output value."""
        return np.asarray(raw_output, dtype=float).reshape([1,1])

    def inference(self, input_length, target_levels, output_levels=None):
        """
        create the dynamic perceptron for sorting models

        :param input_length: the length of the input to be expected for the model
        """
        if output_levels is None:
            output_levels = target_levels

        network_input = tf.placeholder(tf.float32, shape=[1, input_length])

        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable('hidden1_weights',shape=[input_length, 10], initializer=tf.truncated_normal_initializer(stddev=0.04))
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable('hidden1_biases', shape=[10],initializer=tf.constant_initializer(0.1))
            hidden1 = tf.nn.relu(tf.matmul(network_input, weights) + biases, name=scope.name)

        with tf.variable_scope('hidden2') as scope:
            weights = tf.get_variable('hidden2_weights',shape=[10,10], initializer=tf.truncated_normal_initializer(stddev=0.04))
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable('hidden2_biases', shape=[10],initializer=tf.constant_initializer(0.1))
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases, name=scope.name)

        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_weights',shape=[10, output_levels], initializer=tf.truncated_normal_initializer(stddev=1.0))
            biases = tf.get_variable('biases', shape=[target_levels], initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(hidden2, weights), biases, name=scope.name)

        return network_input, softmax_linear


    def loss(self, logits, input_length):
        """
        Add loss to all the trainable variables.
        Args:
            logits: Logits from inference().
            reinforcement_penalties: reinforcement learning loss placeholder
                (should be a tensor)
        Returns:
            Loss tensor of type float.
        """
        reinforcement_penalties = tf.placeholder(tf.float32, shape=[1, 1])
        # Calculate the average reinforcement loss across the batch.
        reinforcement_loss = tf.reduce_mean(tf.matmul(reinforcement_penalties, logits), name='cross_entropy')
        tf.add_to_collection('losses', reinforcement_loss)

        # The total loss is defined as the reinforcement loss plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), reinforcement_penalties

    def train(self, total_loss, global_step, initial_learning_rate=0.1, num_epochs_per_decay=100, learning_rate_decay_factor=0.9, moving_average_decay=0.99):
        """
        :param initial_learning_rate: Learning rate of perceptron
        :param num_epochs_per_decay: number of epochs to pass before decaying
            learning rate
        :param learning_rate_decay_factor: How much the learning rate is
            preserved. Exponent in the exponential decay.
        :param moving_average_decay: *fudge factor to jump directly to minimum
        """

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(initial_learning_rate, global_step, num_epochs_per_decay, learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Compute gradients.
        with tf.control_dependencies([total_loss]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        #variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        #with tf.control_dependencies([apply_gradient_op]):
        #    variables_averages_op = variable_averages.apply(tf.trainable_variables())
        return apply_gradient_op
        #return variables_averages_op

    def update(self, input_vectors, shadho_output):
        """
        :param input_vectors: feature represenation of model results
        :param shadho_output: runtime for each model vecotr of different values
            OR average for all where vector is same length but all the same
            # do greedy first, so first one.
        """
        input_vectors = self.handle_input(input_vectors)
        for input_vector, output_vector,  in zip(input_vectors, shadho_output):
            # optional: transform the shadho_output in some way
            #output_vector = self.transform_shadho_output(output_vector, . . . )
            output_vector = self.handle_output(output_vector)
            self.sess.run(self.train_op, feed_dict={self.reinforcement_penalties: output_vector, self.network_input: input_vector})

    def predict(self, input_vectors):
        input_vectors = self.handle_input(input_vectors)
        logit_list = []
        for input_vector in input_vectors:
            Tracer()()
            logit_list.append(self.sess.run(self.softmax_linear, feed_dict = {self.network_input : input_vector}))
        # outputs log probabilities, convert to non-log probabilities
        logit_list = [np.e ** logits / np.sum(np.e**logits) for logits in logit_list]
        return logit_list,  self.generate_schedule(input_vectors, logit_list)

    def generate_schedule(self, input_vectors, logit_list):
        # TODO deterministic decision
        # top 2 models per ccs
        return {np.where(input_vectors[i][0:len(self.model_ids)])[0][0]:self.compute_class_ids[np.argsort(l)[-2:]] for i, l  in enumerate(logit_list)}

    def close():
        self.sess.close()
