"""
Online reinforcement learning perceptron using mini-batching
"""
import tensorflow as tf

def handle_shadho_sample(shadho):
    # TODO get single sample from shadho and get a the full minibatch
    # then incremental train and predict
    # mini batch so feed fwd with update_freq samples, then update & predict

    # sample = model_id, [1-hot vector size compute classes], cores, avg_cores, memory, virtual_memory, swap_memory, bytes_recieved, bytes_sent
    # expected output: 1d array of probs for each compute class
    # then deterministic decision making process, ie. assign to top 2.

class Perceptron(object):
    """
    Online reinforcement learning perceptron for mapping models to compute
    classes
    """
    def __init__(self, input_length, compute_class_ids, *args, *kwargs):
        self.compute_class_ids = np.array(compute_class_ids)

        self.network_input, self.softmax_linear = inference(input_length)
        self.total_loss, self.reinforcement_penalties= loss(softmax_linear, reinforcement_penalties)
        self.train_op = train(total_loss)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess = tf.Session()
        self.sess.run(init_op)

    def inference(self, input_length):
        """
        create the dynamic perceptron for sorting models

        :param input_length: the length of the input to be expected for the model
        """
        network_input = tf.placeholder(tf.float32, shape=[input_length])

        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable('hidden1_weights',shape=[input_length, 10], initializer=tf.truncated_normal_initializer(stddev=0.04))
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss'))
            tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable('hidden1_biases', shape=[10],initializer=tf.constant_initializer(0.1))
            hidden1 = tf.nn.relu(tf.matmul(network_input, weights) + biases, name=scope.name)

        with tf.variable_scope('hidden2') as scope:
            weights = tf.get_variable('hidden2_weights',shape=[10,10], initializer=tf.truncated_normal_initializer(stddev=0.04))
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss'))
            tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable('hidden2_biases', shape=[10],initializer=tf.constant_initializer(0.1))
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases, name=scope.name)


        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_weights',shape=[10, flags.OUTPUT_LEVELS], initializer=tf.truncated_normal_initializer(stddev=1.0))
            biases = tf.get_variable('biases', shape=[args.target_levels], initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(hidden2, weights), biases, name=scope.name)

        return network_input, softmax_linear


    def loss(self, logits, reinforcement_penalties):
        """
        Add loss to all the trainable variables.
        Args:
            logits: Logits from inference().
            reinforcement_penalties: reinforcement learning loss placeholder
                (should be a tensor)
        Returns:
            Loss tensor of type float.
        """
        reinforcement_penalties = tf.placeholder(tf.float32, shape=[input_length])
        # Calculate the average reinforcement loss across the batch.
        reinforcement_loss = tf.reduce_mean(tf.matmul(reinforcement_penalties, logits), name='cross_entropy')
        tf.add_to_collection('losses', reinforcement_loss)

        # The total loss is defined as the reinforcement loss plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), reinforcement_penalties

    def train(total_loss, global_step, initial_learning_rate=0.1, num_epochs_per_decay=100, learning_rate_decay_factor=0.9, moving_average_decay=0.99):
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
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

        return variables_averages_op

    def update(self, input_vectors, shadho_output):
        """
        :param input_vectors: feature represenation of model results
        :param shadho_output: runtime for each model vecotr of different values
            OR average for all where vector is same length but all the same
            # do greedy first, so first one.
        """
        for output_vector in shadho_output:
            # optional: transform the shadho_output in some way
            #output_vector = self.transform_shadho_output(output_vector, . . . )
            self.sess.run(self.train_op, feed_dict={self.reinforcement_penalty_handle: output_vector, self.input_handle: input_vector})

    def predict(self, input_vectors):
        logit_list = []
        for input_vector in input_vectors:
            logit_list.append(self.sess.run(logits, feed_dict = {self.input_handle : input_vector}))
        # outputs log probabilities, convert to non-log probabilities
        logit_list = [logits.apply(lambda x: np.e ** x / np.sum(np.e**x)), axis=1) for logits in logit_list]
        return logit_list,  self.generate_schedule(input_vectors, logit_list)

    def generate_schedule(self, logit_list)
        # TODO deterministic decision
        # top 2 models per ccs
        return {input_vectors[i][0]:comput_class_idx[np.argsort(l)[-2:]] for i, l  in enumerate(logit_list)}

    def close():
        self.sess.close()
