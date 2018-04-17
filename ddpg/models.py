import tensorflow as tf
import tensorflow.contrib.layers as layers


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            convs=[(64, 8, 4), (32, 4, 2), (32, 3, 1)]
            hiddens=[256]
            out = obs
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                            num_outputs=num_outputs,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            activation_fn=tf.nn.relu)
                conv_out = layers.flatten(out)
            with tf.variable_scope("action_value"):
                action_out = conv_out
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if self.layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=self.nb_actions, activation_fn=None)

            # x = tf.layers.dense(x, 64)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            
            # x = tf.layers.dense(x, 64)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            
            # x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(action_scores)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            convs=[(64, 8, 4), (32, 4, 2), (32, 3, 1)]
            hiddens=[256]
            out = obs
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                            num_outputs=num_outputs,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            activation_fn=tf.nn.relu)
                conv_out = layers.flatten(out)
            with tf.variable_scope("action_value"):
                action_out = conv_out
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if self.layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
            x = tf.nn.relu(action_out)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
