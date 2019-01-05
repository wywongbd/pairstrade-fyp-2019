import tensorflow as tf

# fixed number of time steps in one episode
trading_period = 60

# 1 is zscore, the other 3 is one-hot encoding of the current postion of the trading algorithm
state_dim = 1+3

# RNN hidden state dimension
h_dim = 20

# number of actions
a_num = 4

# number of layer1 output
layer1_out_num = 100

# learning rate
lr = 1e-3

# update batch size
batch_size = 20


# start building compution graph
tf.reset_default_graph()

# policy network
o = tf.placeholder(tf.float32, [None, h_dim] , name="observations")

layer1 = tf.layers.Dense(units=layer1_out_num,
                         activation=tf.keras.layers.LeakyReLU(),
                         kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )(o)

scores = tf.layers.Dense(units=a_num,
                         activation=tf.keras.layers.LeakyReLU(),
                         kernel_initializer=tf.contrib.layers.xavier_initializer()
                        )(layer1)

# train only the weights above
t_vars = tf.trainable_variables()

# for sampling an action during a episode
action_probs = tf.nn.softmax(scores)

# chosen actions
input_actions = tf.placeholder(tf.int32, [None], name="action_label")
advantages = tf.placeholder(tf.float32, [None], name="adjusted_reward_signal")

neg_log_select_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=input_actions)
loss = tf.reduce_mean(neg_log_select_prob * advantages)
grads = tf.gradients(loss, t_vars)

accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
               for tv in t_vars]

reset_grads = [grad.assign(tf.zeros_like(grad))
               for grad in accum_grads]

evaluate_batch = [accum_grad.assign_add(grad/batch_size)
                  for accum_grad, grad in zip(accum_grads, grads)]

adam = tf.train.AdamOptimizer(learning_rate=lr)
apply_grads = adam.apply_gradients(zip(accum_grads, t_vars))

# # output the default graph which can be viewed on tensorboard
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()

