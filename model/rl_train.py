import sys
sys.path.append("../log_helper")

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import random
import argparse
import time
import logging

from os.path import isfile, join, splitext
from datetime import datetime, timedelta
from pytz import timezone, utc

plt.rcParams["font.size"] = 16
plt.rcParams["patch.force_edgecolor"] = True

from process_raw_prices import *
import trading_env
import rl_load_data
import rl_constants
from log_helper import LogHelper

tf.enable_eager_execution()


########################## functions ##############################

def sample_action(logits, batch_size, random=False):
    if random:
        dist = tf.distributions.Categorical(logits=tf.zeros([batch_size, a_num]))
    else:
        dist = tf.distributions.Categorical(logits=logits)
    
    # 1-D Tensor where the i-th element correspond to a sample from
    # the i-th categorical distribution
    return dist.sample()


def discount_rewards(r):
    """
    r is a numpy array in the shape of (n, batch_size).
    
    return the discounted and cumulative rewards"""
    
    result = np.zeros_like(r, dtype=float)
    n = r.shape[0]
    sum_ = np.zeros_like(r[0], dtype=float)
    for i in range(n-1,-1,-1):
        sum_ *= gamma
        sum_ += r[i]
        result[i] = sum_
    
    return result


def loss(all_logits, all_actions, all_advantages):
    neg_log_select_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_logits, labels=all_actions)
    
    # 0 axis is the time axis. 1 axis is the batch axis
    return tf.reduce_mean(neg_log_select_prob * all_advantages, 0)


def extract_pair_name(s):
    return '_'.join(s.split('-')[:2])


def extract_pair_index(s):
    return int(s.split('-')[-1])


def save_model():
    hkg_time = get_hkg_time()
    checkpoint_name = hkg_time.strftime("%Y%m%d_%H%M%S")
    # change the dir name to separate different models...
    checkpoint_prefix = checkpoint_dir+checkpoint_name+"/"
    path = root.save(checkpoint_prefix)
    _logger.info('checkpoint path: {}'.format(path))


def restore_model(checkpoint_dir):
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))


def get_hkg_time():
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Hong_Kong")
    hkg_time = utc_dt.astimezone(my_tz)
    return hkg_time


def run_batch_for_evaluate_performance(return_list, data_indices):
    done = False
    s = env.reset(data_indices)
#     print('portfolio val:', env.port_val[0])

    # for accumalting episode statistics
    act_batch_size = tf.shape(s).numpy()[0]
    total_r = np.zeros(act_batch_size)

    # internally the episode length is fixed by trading_period
    while not done:
        logits = pi(s)
        a = sample_action(logits, act_batch_size)

        # get immediate reward, update state, and get done
        r, s, done = env.step(a.numpy())
        

#         # for debugging
#         print('logits:', logits)
#         print('a:', a.numpy())
#         print('r:', r)
#         print('s:', s)
#         print('portfolio val:', env.port_val[0])

    total_r = env.port_val_minus_com-rl_constants.initial_cash
    return_list += total_r.tolist()
#     return {extract_pair_name(trading_env.curr_pairs[i]): total_r[i] for i in range(act_batch_size)}
    return {trading_env.curr_pairs[i]: total_r[i] for i in range(act_batch_size)}


def run_epoch_for_evaluate_performance(data_indices):
    rs = []
    total_r_dict = {}
    trading_env.glob_indices = None # reset the dataset
    counter = 0
    temp_dict = run_batch_for_evaluate_performance(rs, data_indices)
    total_r_dict.update(temp_dict)
    _logger.info('{}, '.format(counter))
    counter += 1
    while trading_env.glob_indices != None:
        temp_dict = run_batch_for_evaluate_performance(rs, data_indices)
        total_r_dict.update(temp_dict)
        _logger.info('{}, '.format(counter))
        counter += 1
    return rs, total_r_dict


def plot_rs_dist(rs, fig_name, fig_title):
    return_in_percent = np.array(rs) / rl_constants.initial_cash
    plt.figure()
    stat = plt.hist(return_in_percent, bins=30)
    
    mean = return_in_percent.mean()
    median = np.median(return_in_percent)
    
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean: {:.3f}'.format(mean))
    plt.axvline(median, color='y', linestyle='dashed', linewidth=1, label='Median: {:.3f}'.format(median))
    
    plt.gcf().set_size_inches(14, 7)
    
    if len(fig_title) != 0:
        plt.suptitle(fig_title)
    plt.xlabel('return')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    _logger.info(stat)
    plt.savefig(join(plot_folder_path, fig_name+'.png'))
#     plt.close()
    _logger.info('Number of pairs: {}'.format(len(return_in_percent)))
    _logger.info('Mean return over all pairs: {:.4f}'.format(np.mean(return_in_percent)))

    
# global batch_id to keep track of the progress
batch_no = 0
best_average_return_within_epoch = 0.0
def train(data_indices, num_of_batch):
    global batch_no
    global best_average_return_within_epoch
    # print parameters
    _logger.info('num_of_pair = {}'.format(rl_constants.num_of_pair))
    _logger.info('batch_size = {}'.format(batch_size))
    _logger.info('num_of_batch = {}, estimated epoch = {}'.format(num_of_batch, num_of_batch*batch_size/2/rl_constants.num_of_pair))
    _logger.info('rand_action_prob = {}'.format(rand_action_prob))
    _logger.info('lr = {}'.format(lr))
    trading_env.glob_indices = None # reset the dataset

    # for training reference only
    average_return_within_batch = 0.0
    average_return_within_epoch = 0.0
    num_eps_over = 0
    num_eps_over_for_epoch = 0
#     total_r_dict = {}

    start_time = time.time()
    for batch in range(num_of_batch):

        with tf.GradientTape() as gt:
            # saving for update
            all_logits = []
            all_actions = []
            all_rewards = []

            # episode starts here~
            done = False
            s = env.reset(data_indices)

            # for accumalting episode statistics
            act_batch_size = tf.shape(s).numpy()[0]
            num_eps_over += act_batch_size
            num_eps_over_for_epoch += act_batch_size
            total_r = np.zeros(act_batch_size)

            # internally the episode length is fixed by trading_period
            while not done:
                logits = pi(s)
                a = sample_action(logits, act_batch_size, random=np.random.rand() <= rand_action_prob)
                r, s, done = env.step(a.numpy())

                # save the episode
                all_logits.append(logits)
                all_actions.append(a)
                all_rewards.append(r)

#                 r_sum = np.sum(r)
#                 average_total_r += r_sum
#                 epoch_average_total_r += r_sum
#                 total_r += r

    #             # debugging
    #             print(env.t)
    #             print(env.t+1==200)
    #             print(r[0])
    #             print('a:', a.numpy())
    #             print(done)
    #             print(logits)

            # keep track of the pair performance (of course this is not totally fair for all pairs
            # as there are parameters update).
#             total_r_dict.update({curr_pairs[i]: total_r[i] for i in range(act_batch_size)})
            average_return_within_batch += np.sum(env.port_val)
            average_return_within_epoch += np.sum(env.port_val)

            all_logits_stack = tf.stack(all_logits)
            all_actions_stack = tf.stack(all_actions)
            all_rewards_stack = np.array(all_rewards)

            # compute cummulative rewards for each action
            all_cum_rewards = discount_rewards(all_rewards_stack)
            all_cum_rewards -= np.mean(all_cum_rewards)
    #         all_cum_rewards /= np.std(all_cum_rewards)
    #         all_cum_rewards /= np.mean(np.abs(all_cum_rewards))
            all_cum_rewards /= rl_constants.initial_cash
            all_cum_rewards = tf.convert_to_tensor(all_cum_rewards, dtype=tf.float32)

            loss_value = loss(all_logits_stack, all_actions_stack, all_cum_rewards)

        grads = gt.gradient(loss_value, state_encoding_model.variables + pi.variables)
        optimizer.apply_gradients(zip(grads, state_encoding_model.variables + pi.variables))

        if (batch_no+1) % batches_per_print == 0:
            end_time = time.time()
            
            average_return_within_batch /= (num_eps_over)
            average_return_within_batch -= rl_constants.initial_cash
            average_return_within_batch /= rl_constants.initial_cash
            
            _logger.info(("batch_id: {}, num_eps_over: {}, average_return_per_ep: {:.4f}, "+
                   "time_spent: {:.1f}s").format(
                batch_no, num_eps_over, average_return_within_batch, end_time-start_time))

            # reset
            average_return_within_batch = 0.0
            num_eps_over = 0
            start_time = time.time()

        # print epoch summary
        if trading_env.glob_indices == None:
            
            average_return_within_epoch /= num_eps_over_for_epoch
            average_return_within_epoch -= rl_constants.initial_cash
            average_return_within_epoch /= rl_constants.initial_cash
            
            # compute average total reward in one epoch to evaluate agent performance
            _logger.info("average total_r over one epoch: {:.4f}".format(
                average_return_within_epoch))
            
            
            if best_average_return_within_epoch < average_return_within_epoch:
                save_model()
                best_average_return_within_epoch = average_return_within_epoch

            # reset
            average_return_within_epoch = 0.0
            num_eps_over_for_epoch = 0
#             total_r_dict = {}

        batch_no += 1

    _logger.info('Finished training~')

########################## functions ##############################


myLeakyReLU = tf.keras.layers.LeakyReLU()
myLeakyReLU.__name__ = "myLeakyReLU"


# classes
class TradingPolicyModel(tf.keras.Model):
    def __init__(self):
        super(TradingPolicyModel, self).__init__()
        self.dense1 = tf.layers.Dense(units=layer1_out_num,
                                      activation=myLeakyReLU,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)
                                     )
        self.dense2 = tf.layers.Dense(units=layer1_out_num,
                                      activation=myLeakyReLU,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)
                                     )
#         self.dense3 = tf.layers.Dense(units=layer1_out_num,
#                                       activation=myLeakyReLU,
#                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)
#                                      )
#         self.dense4 = tf.layers.Dense(units=layer1_out_num,
#                                       activation=tf.keras.layers.LeakyReLU(),
#                                       kernel_initializer=tf.contrib.layers.xavier_initializer()
#                                      )
        self.logits = tf.layers.Dense(units=rl_constants.a_num,
                                      activation=myLeakyReLU,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(reg)
                                     )

    def call(self, inputs):
        # Forward pass
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)
#         inputs = self.dense3(inputs)
#         inputs = self.dense4(inputs)
        logits = self.logits(inputs)
        return logits


class StateEncodingModel(tf.keras.Model):
    def __init__(self, batch_size, num_rnn_layers):
        super(StateEncodingModel, self).__init__()
        self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(h_dim) for i in range(num_rnn_layers)])
        self.reset_state(batch_size)
    
    def call(self, inputs):
        output, self.state = self.cell(inputs, self.state)
        return output
        
    def reset_state(self, batch_size):
        self.state = self.cell.zero_state(batch_size, tf.float32)

        
def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True, help="The job name.")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--num_of_epoch", type=int, default=80)
    parser.add_argument("--h_dim", type=int, default=300, help="RNN hidden state dimension")
    parser.add_argument("--num_rnn_layers", type=int, default=1, help="number of RNN layer")
    parser.add_argument("--layer1_out_num", type=int, default=20, help="number of layer1 output")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--reg", type=float, default=0.00001)
    parser.add_argument("--gamma", type=float, default=5e-4, help="discount factor in reinforcement learning")
    parser.add_argument("--rand_action_prob", type=float, default=0.0, help="random action probability lower is better...")
    parser.add_argument("--batches_per_print", type=int, default=10*5)
    parser.add_argument("--train_indices", default=[0], nargs='+', type=int,
                        help="a list of int that indicate the training period. valid range: 0<=indices<=3.")
    parser.add_argument("--test_indices", default=[1], nargs='+', type=int,
                        help="a list of int that indicate the training period.")
    return parser


job_name = None
batch_size = None
num_of_epoch = None
num_of_batch = None
h_dim = None
num_rnn_layers = None
layer1_out_num = None
lr = None
reg = None
gamma = None
rand_action_prob = None
batches_per_print = None
def copy_config(config):
    global job_name
    global batch_size
    global num_of_epoch
    global num_of_batch
    global h_dim
    global num_rnn_layers
    global layer1_out_num
    global lr
    global reg
    global gamma
    global rand_action_prob
    global batches_per_print
    
    job_name = config.job_name
    batch_size = config.batch_size
    num_of_epoch = config.num_of_epoch
    num_of_batch = 2*rl_constants.num_of_pair*num_of_epoch//batch_size # pass to train function
    h_dim = config.h_dim
    num_rnn_layers = config.num_rnn_layers
    layer1_out_num = config.layer1_out_num
    lr = config.lr
    reg = config.reg
    gamma = config.gamma
    rand_action_prob = config.rand_action_prob
    batches_per_print = config.batches_per_print


        
pi = None
env = None
state_encoding_model = None
optimizer = None
root = None
plot_folder_path = None
checkpoint_dir = None
_logger = None
def main(config):
    global pi
    global state_encoding_model
    global env
    global optimizer
    global root
    global plot_folder_path
    global checkpoint_dir
    global _logger
    
    copy_config(config)
    
    
    plot_folder_path = './logging/{}/plots/'.format(job_name)
    checkpoint_dir = './logging/{}/saved_models/'.format(job_name)
    log_folder_path = './logging/{}/'.format(job_name)

    os.makedirs(plot_folder_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    LogHelper.setup(log_path=log_folder_path+'log.txt', log_level=logging.INFO)
    _logger = logging.getLogger(__name__)
    
    _logger.info("Hello World!")
    
    _logger.info("config.train_indices = {}".format(config.train_indices))
    _logger.info("config.test_indices = {}".format(config.test_indices))
    
    
    # load data
    all_pairs_slices, all_pairs_df, trading_period = rl_load_data.load_data()


    # create objects
    pi = TradingPolicyModel()
    state_encoding_model = StateEncodingModel(batch_size, num_rnn_layers)
    env = trading_env.TradingEnvironment(
        state_encoding_model,
        all_pairs_slices,
        all_pairs_df,
        trading_period,
        batch_size,
        rl_load_data.col_name_to_ind
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # create checkpoint object
    root = tf.train.Checkpoint(pi=pi, state_encoding_model=state_encoding_model, optimizer=optimizer)
    
    # evaluate performance on train dataset
    train_rs, train_total_r_dict = run_epoch_for_evaluate_performance(config.train_indices)
    plot_rs_dist(train_rs, 'RL_train_result_before_train', '')
    
    # evaluate performance on test dataset
    test_rs, test_total_r_dict = run_epoch_for_evaluate_performance(config.test_indices)
    plot_rs_dist(test_rs, 'RL_test_result_before_train', '')
    
    train(config.train_indices, 300)
#     train(config.train_indices, num_of_batch)
    
    # evaluate performance on train dataset
    train_rs, train_total_r_dict = run_epoch_for_evaluate_performance(config.train_indices)
    plot_rs_dist(train_rs, 'RL_train_result_after_train', '')
    
    # evaluate performance on test dataset
    test_rs, test_total_r_dict = run_epoch_for_evaluate_performance(config.test_indices)
    plot_rs_dist(test_rs, 'RL_test_result_after_train', '')

        
if __name__ == '__main__':
    
    config = generate_parser().parse_args()
    
    main(config)
