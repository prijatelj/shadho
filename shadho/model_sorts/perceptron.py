"""
Online reinforcement learning perceptron using mini-batching
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from IPython.core.debugger import Tracer

class Perceptron(object):
    """
    Online reinforcement learning perceptron for mapping models to compute
    classes
    """
    def __init__(self, input_length, target_levels, model_ids, compute_class_ids, output_levels=None, decay_lambda=0.85, reinit_decay_lambda=None, epsilon=0.1, top_n=1, *args, **kwargs):
        self.input_length = input_length
        self.top_n = top_n
        self.epsilon = epsilon # epsilon for reinforcement learning decision making

        # Create queue structure for circular queue of predictions.
        self.pred_queue = {cc_id:{'queue':None, 'idx':None, 'looped':None} for cc_id in compute_class_ids}

        self.model_ids = np.array(model_ids)
        self.compute_class_ids = np.array(compute_class_ids)

        self.decay_lambda = decay_lambda # global decay factor for all moving averages
        #self.reinit_decay_lambda = decay_lambda * . if reinit_decay_lambda is None else reinit_decay_lambda
        self.reinit_decay_lambda = 0.74
        #self.reinit_decay_lambda = 0.84
        self.reinit_counter = 0
        self.reinit_strikes = 0


        self.param_averages = {m_id:None for m_id in model_ids}
        self.time_averages = {m_id:None for m_id in model_ids}
        self.reinit_time_averages = {m_id:None for m_id in model_ids}

        # placeholder for normalization factors of the non-one_hot features.
        self.normalize_factors = None

        self.network_input, self.softmax_linear = self.inference(input_length, target_levels, output_levels)
        self.total_loss, self.reinforcement_penalties= self.loss(self.softmax_linear, input_length, target_levels)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op, self.grads = self.train(self.total_loss, self.global_step)
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def is_pred_queue_empty(self, compute_class_id):
        return self.pred_queue[compute_class_id]['queue'] is None

    @property
    def any_pred_queue_empty(self):
        return any([self.is_pred_queue_empty(cc_id) for cc_id in self.compute_class_ids])

    def next_pred(self, compute_class_id):
        """Returns current value and moves pointer to next index."""
        pred = self.pred_queue[compute_class_id]['queue'][self.pred_queue[compute_class_id]['idx']] if self.pred_queue[compute_class_id]['queue'] else None

        print('next_pred cc_id', compute_class_id)
        print('next_pred queue', self.pred_queue[compute_class_id]['queue'])
        print('next_pred idx', self.pred_queue[compute_class_id]['idx'])
        print('next_pred idx', self.pred_queue[compute_class_id]['looped'])
        if self.pred_queue[compute_class_id]['idx'] >= len(self.pred_queue[compute_class_id]['queue']) - 1:
            self.pred_queue[compute_class_id]['idx'] = 0
            if not self.pred_queue[compute_class_id]['looped']:
                self.pred_queue[compute_class_id]['looped'] = True
        else:
            self.pred_queue[compute_class_id]['idx'] += 1

        return pred

    def update_pred_queue(self, preds):
        """Sets pred_queue to provided list of predictions."""
        # NOTE would be easier if it was cc to model in the fisrt place.
        # loop through all predictions and assign them to proper queue.
        for pred in preds:
            model_id = pred[0]
            for pred_cc in pred[1:]:
                # if queue exists and has not fully looped, add to existing queue
                if self.pred_queue[pred_cc]['queue'] and not self.pred_queue[pred_cc]['looped']:
                    if self.pred_queue[pred_cc]['idx'] != 0:
                        del self.pred_queue[pred_cc]['queue'][:self.pred_queue[pred_cc]['idx']]
                        self.pred_queue[pred_cc]['idx'] = 0

                    self.pred_queue[pred_cc]['queue'].append(model_id)
                else: # if DNE or looped, create the queue, reset idx and looped
                    self.pred_queue[pred_cc]['queue'] = [model_id]
                    self.pred_queue[pred_cc]['idx'] = 0
                    self.pred_queue[pred_cc]['looped'] = False

    def model_id_to_onehot(self, model_id):
        return (self.model_ids == model_id).astype(float)
        #return (self.model_ids == model_id).astype(int)

    def compute_class_to_onehot(self, compute_class):
        return (self.compute_class_ids == compute_class).astype(float)
        #return (self.compute_class_ids == compute_class).astype(int)

    def set_normalize_factors(self, shadho_backend, resources, weights=None):
        """
        Given SHADHO backend and list of resources, get max of each resource
        feature.
        """
        # NOTE set normalize factors after initial run always.
        #normals_count = len(self.sess.run(self.network_input)) - (len(self.model_ids) + len(self.compute_class_ids))
        if weights is None:
            normalize_factors = np.ones(len(resources))
            # find the max resource values across all models for initial run
            for model_id in shadho_backend.models: # every model
                for i, resource_id in enumerate(resources): # each resource
                    # iterate through model history, which == len(compute_class_ids)
                    for j in range(len(self.compute_class_ids)): # history
                        if shadho_backend.models[model_id].results[-j].results is not None:
                            normalize_factors[i] = np.maximum(shadho_backend.models[model_id].results[-j].results['resources_measured'][resource_id], normalize_factors[i])
            self.normalize_factors = normalize_factors.astype(float)
        else:
            self.normalize_factors = np.asarray(weights, dtype=float)

    def normalize_input(self, input_vector, normalize_factors=None):
        """
        Given 1D array of length non-1hot features, divide those features by
        their corresponding factor.
        """
        one_hot_size = len(self.model_ids) + len(self.compute_class_ids)

        if normalize_factors is None and self.normalize_factors is not None:
            normalize_factors = self.normalize_factors

        # assumes that there are more features than the 1 hot vectors
        #if self.input_length <= one_hot_size:
        #    return input_vector
        #    #elif len(input_vector) != self.input_length: # should throw error
        #    #    return input_vector
        #else:
        input_vector = input_vector.astype(float)
        input_vector[:, one_hot_size:] = input_vector[:, one_hot_size:] / normalize_factors

        return input_vector

    def handle_input(self, raw_input_vectors):
        """
        Handles multiple raw input vectors.
        :return: changes the list of raw input into param averages, convert the input, or None if param averages is None.
        """
        input_vectors = []
        for i in raw_input_vectors:
            if i[1] is None: # if failed results exists, run average prediction.
                input_vectors.append(self.param_averages[i[0]])
                print('self.param_averages[i[0]]', self.param_averages[i[0]])
            else:
                input_vector = np.append(np.append(self.model_id_to_onehot(i[0]), self.compute_class_to_onehot(i[1])),  i[2:]).reshape([1, -1]).astype(float)
                input_vectors.append(self.normalize_input(input_vector))
                # TODO possibly add custom error for caller to handle.
        return input_vectors

    def handle_output(self, raw_output):
        """Handles single raw_output value."""
        return np.asarray(raw_output, dtype=float)

    def inference(self, input_length, target_levels, output_levels=None):
        """
        create the dynamic perceptron for sorting models

        :param input_length: the length of the input to be expected for the model
        """
        if output_levels is None:
            output_levels = target_levels

        network_input = tf.placeholder(tf.float32, shape=[1, input_length])

        with tf.variable_scope('hidden1') as scope:
            weights = tf.get_variable('hidden1_weights',shape=[input_length, output_levels], initializer=tf.truncated_normal_initializer(stddev=0.5))
            biases = tf.get_variable('hidden1_biases', shape=[output_levels],initializer=tf.constant_initializer(0.1))
            hidden1 = tf.nn.sigmoid(tf.matmul(network_input, weights) + biases, name=scope.name)

        # Deeper network components currently removed; further testing and alteration of loss probably required to use them
        # Also removed all of the L2 loss terms becuase they were screwing with training and I didn't feel like normalizing the loss to fix that

        # with tf.variable_scope('hidden2') as scope:
        #     weights = tf.get_variable('hidden2_weights',shape=[10,output_levels], initializer=tf.truncated_normal_initializer(stddev=0.5))
        #     # weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        #     # tf.add_to_collection('losses', weight_decay)
        #     biases = tf.get_variable('hidden2_biases', shape=[output_levels],initializer=tf.constant_initializer(0.1))
        #     hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases, name=scope.name)

        # with tf.variable_scope('softmax_linear') as scope:
        #     weights = tf.get_variable('softmax_weights',shape=[10, output_levels], initializer=tf.truncated_normal_initializer(stddev=1.0))
        #     biases = tf.get_variable('biases', shape=[target_levels], initializer=tf.constant_initializer(0.0))
        #     softmax_linear = tf.add(tf.matmul(hidden2, weights), biases, name=scope.name)

        return network_input, hidden1

    def loss(self, logits, input_length, target_levels):
        """
        Add loss to all the trainable variables.
        Args:
            logits: Logits from inference().
            reinforcement_penalties: reinforcement learning loss placeholder
                (should be a tensor)
        Returns:
            Loss tensor of type float.
        """
        reinforcement_penalties = tf.placeholder(tf.float32, shape=[1, target_levels])
        # Calculate the reinforcement loss.
        reinforcement_loss = tf.squeeze(tf.matmul(reinforcement_penalties, -tf.log(logits + (0.001)),transpose_b=True,name='rl_product'), name='rl_loss')
        tf.add_to_collection('losses', reinforcement_loss)

        # The total loss is defined as the reinforcement loss plus all of the weight decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), reinforcement_penalties

    def train(self, total_loss, global_step, initial_learning_rate=0.8, num_epochs_per_decay=160, learning_rate_decay_factor=1.0, moving_average_decay=0.99):
        """
        :param initial_learning_rate: Learning rate of perceptron
        :param num_epochs_per_decay: number of epochs to pass before decaying
            learning rate
        :param learning_rate_decay_factor: How much the learning rate is
            preserved. Exponent in the exponential decay.
        :param moving_average_decay: *fudge factor to jump directly to minimum
        """

        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.train.exponential_decay(initial_learning_rate, global_step, num_epochs_per_decay, learning_rate_decay_factor, staircase=True)

        # Use a fixed learning rate
        lr = initial_learning_rate

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
        #return variables_averages_op

        return apply_gradient_op, grads

    def update(self, input_vectors, shadho_output):
        """
        :param input_vectors: feature represenation of model results
        :param shadho_output: runtime for each model vecotr of different values
            OR average for all where vector is same length but all the same
            # do greedy first, so first one.
        """
        models = [x[0] for x in input_vectors]
        cc_ids = [x[1] for x in input_vectors]
        print('input vec')
        for input_vec in input_vectors:
            print(input_vec)
        print('compute classes:')
        for cc in self.compute_class_ids:
            print(cc)
        input_vectors = self.handle_input(input_vectors)
        print('handled input vec')
        for input_vec in input_vectors:
            print(input_vec)

        for input_vector, output_vector, model, cc  in zip(input_vectors, shadho_output, models, cc_ids):
            output_vector = self.handle_output(output_vector)

            if self.param_averages[model] is None:
                self.param_averages[model] = input_vector
                self.time_averages[model] = output_vector
            else:
                self.param_averages[model] = input_vector * (1-self.decay_lambda) + self.param_averages[model] * self.decay_lambda
                self.time_averages[model] = output_vector * (1-self.decay_lambda) + self.time_averages[model] * self.decay_lambda

                self.reinit_time_averages[model] = output_vector * (1-self.reinit_decay_lambda) + self.time_averages[model] * self.reinit_decay_lambda
                self.reinit_counter += 1

                # TODO reinit after time to converge, recent loss is significantly worse recently
                #if self.sess.run(self.global_step) > 1000 == 0 \
                if self.reinit_counter > 2500 \
                     and np.mean(list(self.reinit_time_averages.values())) > 1 * np.mean(list(self.time_averages.values())):
                    if self.reinit_strikes > 5:
                        self.reinit()
                        self.reinit_counter = 0
                        self.reinit_strikes = 0
                    else:
                        self.reinit_strikes += 1
                elif self.reinit_strikes > 0:
                    self.reinit_strikes -= 1


            rl_vector = (np.sign((self.time_averages[model] - output_vector)/self.time_averages[model] - 0.005) * self.compute_class_to_onehot(cc).reshape(1,-1))

            self.sess.run(self.train_op, feed_dict={self.reinforcement_penalties: rl_vector, self.network_input: self.param_averages[model]})
            #print(f"rl_vector: {rl_vector} | post_losses: {self.sess.run(tf.get_collection('losses'), feed_dict={self.reinforcement_penalties: rl_vector, self.network_input: input_vector})}")

    def predict(self, input_vectors):
        print('predict: input vec:')
        for input_vec in input_vectors:
            print(input_vec)

        models = [x[0] for x in input_vectors]

        input_vectors = self.handle_input(input_vectors)
        print('predict handled input vec:')
        for input_vec in input_vectors:
            print(input_vec)

        logit_list = []
        random_guess = []

        # loop through input provided and make predictions
        for i, model_id in enumerate(models):
            if self.param_averages[model_id] is not None:
                # use the averages if exists
                    logit_list.append(self.sess.run(self.softmax_linear, feed_dict = {self.network_input : self.param_averages[model_id]}))
            elif input_vectors[i] is not None: #TODO check if valid input
                # use the actual sample
                logit_list.append(self.sess.run(self.softmax_linear, feed_dict = {self.network_input : input_vectors[i]}))
            else:
                # No input information so random guess
                random_guess.append(np.append(model_id, np.random.choice(self.compute_class_ids, size=self.top_n, replace=False)))

        # outputs log probabilities, convert to non-log probabilities
        logit_list = [np.e ** logits / np.sum(np.e**logits) for logits in logit_list]
        logit_list = [np.squeeze(logits) for logits in logit_list]

        preds = self.generate_schedule(input_vectors, logit_list)
        if len(random_guess) > 0:
            # append random_guess to predictions.
            preds += random_guess
        self.update_pred_queue(preds)
        return logit_list, preds

    def generate_schedule(self, input_vectors, logit_list):
        """
        :return: list of lists: [model_id, cc_id1, cc_id2, ...] The cc_ids for each model to go.
        """
        # NOTE Could just update the pred_queue specific to each cc!

        # top 1 models per ccs
        print('input_vectors len = ',len(input_vectors), ' 1st 10 = ', input_vectors[:10])
        print('logit_list len = ', len(logit_list), ' 1st 10 = ', logit_list[:10])
        return [np.append(self.model_ids[np.where(x[0][0:len(self.model_ids)])[0][0]], self.compute_class_ids[np.argsort(y)[::-1][:self.top_n] if (np.random.uniform() > self.epsilon) else np.random.choice(range(len(self.compute_class_ids)))]) for x,y in zip(input_vectors,logit_list)]

    def reinit(self):
        # Re-initialize the learner. Potentially a good idea to do periodically, as it has a bit of trouble climbing out of deep local minima
        self.sess.run(self.init_op)

    def close(self):
        self.sess.close()
