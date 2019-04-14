import random
import numpy as np
import tensorflow as tf
import logging
import sys

sys.path.append("./model")

import rl_load_data
import rl_constants

_logger = logging.getLogger(__name__)

# glob_indices should be assigned to None if an epoch finished
glob_indices = None
sample_start_index = None
list_of_pairs_in_epoch = None
curr_pairs = None
def get_random_history(all_pairs_slices, all_pairs_df, batch_size, indices, pair_name=None):
    """Sample some pairs and get the history of those pairs. The history should have
    three dimension. The first dimension is for time. The second dimension is indexed
    by features name. The third dimension is the index of training instance.
    """
    global glob_indices
    global sample_start_index
    global list_of_pairs_in_epoch
    global curr_pairs
    
    # checking
    if not isinstance(indices, list):
        raise Exception('indices should be a list of index from 0 to {}'.format(rl_constants.num_of_period-1))
    elif len(indices) == 0:
        raise Exception('indices should be non empty.')
    elif False in [(0<=i and i<rl_constants.num_of_period) for i in indices]:
        raise Exception('indices should be a list of index from 0 to {}'.format(rl_constants.num_of_period-1))
    
    if pair_name == None:
        # reset the list of pairs in epoch
        if glob_indices == None or set(glob_indices) != set(indices):
            glob_indices = indices
            list_of_pairs_in_epoch = []
            for i in glob_indices:
                list_of_pairs_in_epoch.extend(all_pairs_slices[i])
            random.shuffle(list_of_pairs_in_epoch)
            sample_start_index = 0

        # normal update of index
        sample_end_index = sample_start_index+batch_size
        sample_pair_slices = list_of_pairs_in_epoch[sample_start_index:sample_end_index]
        curr_pairs = sample_pair_slices

        # reached the end of data
        if sample_end_index >= len(list_of_pairs_in_epoch):
            glob_indices = None

        # update index for next batch
        sample_start_index += batch_size
    else:
        # assume data_indices only has one element
        stock1, stock2 = pair_name.split('-')
        pair_name1 = stock2+"-"+stock1
        pair_df_name = pair_name+"-{}".format(indices[0])
        pair_df_name1 = pair_name1+"-{}".format(indices[0])
        _logger.info("name {}".format(pair_df_name))
        _logger.info("name1 {}".format(pair_df_name1))
        
        _logger.info("keys {}".format([k for k, v in all_pairs_df.items()]))
        
        if pair_df_name in all_pairs_df:
            sample_pair_slices = [pair_df_name]
            
        elif pair_df_name1 in all_pairs_df:
            sample_pair_slices = [pair_df_name1]
        
        else:
            _logger.info("error loading pair {}!".format(pair_name))
    
    # return to the environment. this should be no greater than batch_size
    actual_batch_size = len(sample_pair_slices)
    
    history = []
    for s in sample_pair_slices:
#         df = pd.read_csv(join(dataset_folder_path, s+".csv"))
        df = all_pairs_df[s]
        df_val = df[rl_load_data.df_columns].values
        history.append(df_val)
    
    history = np.array(history)
    return np.transpose(history, (1, 2, 0)), actual_batch_size


def compute_input_history(history):
    """Slicing history in its second dimension."""
    # no slicing for now
    return history[:,2:5]

#     # no spread
#     return history[:,2:4]


def long_portfolio_value(q, p):
    return q*p


def short_portfolio_value(q, p, init_p):
    return q*(3.0*init_p/2 - p)


def incur_commission(price, qty):
    return min(max(1, 0.005*qty), 0.01*price*qty)


class TradingEnvironment():
    """Trading environment for reinforcement learning training.
    
    NOTE: Call reset first before calling step!
    
    Arguments:
        state_encoding_model: the model that encode past input_history data into a state
        vector which will be fed as input to the policy network.
    """
    def __init__(self, state_encoding_model, all_pairs_slices, all_pairs_df, trading_period, batch_size, col_name_to_ind):
        # do some initialization
        self.state_encoding_model = state_encoding_model
        self.trading_period = trading_period
        self.batch_size = batch_size
        self.all_pairs_slices = all_pairs_slices
        self.all_pairs_df = all_pairs_df
        self.col_name_to_ind = col_name_to_ind
        
    def _reset_env(self, data_indices, pair_name):
        
        # prepare a batch of history and input_history
        # actual batch_size depends on the dataset
        self.history, curr_batch_size = get_random_history(self.all_pairs_slices,
                                                           self.all_pairs_df,
                                                           self.batch_size,
                                                           data_indices,
                                                           pair_name)
        batch_size = curr_batch_size
        self.input_history = compute_input_history(self.history)
        
        self.t = 0
        self.state_encoding_model.reset_state(batch_size)

        # 0 is no position. 1 is long the spread. 2 is short the spread
        self.position = np.zeros(batch_size, dtype=int)
        
        # initialize the cash each agent has
        self.cash = np.ones(batch_size)*rl_constants.initial_cash
        self.port_val = np.ones(batch_size)*rl_constants.initial_cash
        self.port_val_minus_com = np.ones(batch_size)*rl_constants.initial_cash
        
        # only useful when there is a postion on the spread
        self.quantity = {'x': np.zeros(batch_size), 'y': np.zeros(batch_size)}
        
        # for compute current portfolio value of the short side
        self.short_side_init_price = np.zeros(batch_size)
        
        # create or update self.state variable
        self.update_state()
    
    def reset(self, data_indices, pair_name=None):
        """Return an initial state for the trading environment"""
        
        # determine what dataset to use
        self._reset_env(data_indices, pair_name)
        return self.state
    
    def compute_reward(self, action):
        """Compute the reward at time t which is the change in total portfolio value
        from time t to t+1. It also update the position for time t+1. Exit trade when
        the short side portfolio value <= 0."""
        
        r = np.zeros_like(action, dtype=float)
        cur_his = self.history[self.t]
        nex_his = self.history[self.t+1]
        
        # compute for each training instance in a batch
        for i, a in enumerate(action):
            y_p = cur_his[self.col_name_to_ind["y_close"], i]
            x_p = cur_his[self.col_name_to_ind["x_close"], i]
            nex_y_p = nex_his[self.col_name_to_ind["y_close"], i]
            nex_x_p = nex_his[self.col_name_to_ind["x_close"], i]
            
            if a == 0: # take no position on the spread at time t (current time step)
                if self.position[i] != 0:
                    # need to exit at current time step
                    self.cash[i] = self.port_val_minus_com[i]
                    self.port_val[i] = self.port_val_minus_com[i]
                    
                # compute reward (no change since no position on the spread)
                r[i] = 0
                
                # record the current situation
                self.position[i] = 0
                self.quantity['y'][i] = 0
                self.quantity['x'][i] = 0
            elif a == 1: # long the spread: long Y and short X
                if self.position[i] == 2:
                    # need to exit at current time step
                    self.cash[i] = self.port_val_minus_com[i]
                
                # quantity of each stock will change when the current position is not previous position
                if self.position[i] != 1:
                    # compute quantity from cash
                    self.quantity['y'][i] = int(2.0*self.cash[i]/3.0/y_p)
                    self.quantity['x'][i] = int(2.0*self.cash[i]/3.0/x_p)
                    self.short_side_init_price[i] = x_p
                    
                    # compute entering commission
                    enter_commission = (incur_commission(y_p, self.quantity['y'][i])
                                       +incur_commission(x_p, self.quantity['x'][i]))
                    
                    # cash remaining after entering a position
                    # initial cash - investment amount and commission
                    self.cash[i] -= (0.5*self.quantity['x'][i]*x_p + self.quantity['y'][i]*y_p
                                     + enter_commission)
                
                lpv = long_portfolio_value(self.quantity['y'][i], y_p)
                spv = short_portfolio_value(self.quantity['x'][i], x_p, self.short_side_init_price[i])
                current_port_val = self.cash[i] + lpv + spv

                lpv_nex = long_portfolio_value(self.quantity['y'][i], nex_y_p)
                spv_nex = short_portfolio_value(self.quantity['x'][i], nex_x_p, self.short_side_init_price[i])
                
                # the zero here can be changed to other positive threshold ...
                if spv_nex <= 0:
                    # we loss all the money in the short side
                    # so need to exit the long side
                    self.port_val_minus_com[i]  = (
                        self.cash[i] + lpv_nex - incur_commission(nex_y_p, self.quantity['y'][i])
                    )
                    
                    # forced to take position 0. this mean all the assets transformed into cash
                    self.position[i] = 0
                    self.quantity['y'][i] = 0
                    self.quantity['x'][i] = 0
                    self.cash[i] = self.port_val_minus_com[i]
                    self.port_val[i] = self.port_val_minus_com[i]
                else:
                    exit_commission = (incur_commission(nex_y_p, self.quantity['y'][i])
                                      +incur_commission(nex_x_p, self.quantity['x'][i]))
                    self.port_val[i] = self.cash[i] + lpv_nex + spv_nex
                    self.port_val_minus_com[i] = self.cash[i] + lpv_nex + spv_nex - exit_commission
                    self.position[i] = 1
                
                r[i] = self.port_val_minus_com[i] - current_port_val
                
            elif a == 2: # short the spread: short Y and long X
                if self.position[i] == 1:
                    # need to exit at current time step
                    self.cash[i] = self.port_val_minus_com[i]
                
                # quantity will change when the current position is not previous position
                if self.position[i] != 2:
                    # compute quantity from cash
                    self.quantity['y'][i] = int(2.0*self.cash[i]/3.0/y_p)
                    self.quantity['x'][i] = int(2.0*self.cash[i]/3.0/x_p)
                    self.short_side_init_price[i] = y_p
                    
                    # compute entering commission
                    enter_commission = (incur_commission(y_p, self.quantity['y'][i])
                                       +incur_commission(x_p, self.quantity['x'][i]))
                    
                    # cash remaining after entering a position
                    # initial cash - investment amount and commission
                    self.cash[i] -= (self.quantity['x'][i]*x_p + 0.5*self.quantity['y'][i]*y_p
                                     + enter_commission)
                
                lpv = long_portfolio_value(self.quantity['x'][i], x_p)
                spv = short_portfolio_value(self.quantity['y'][i], y_p, self.short_side_init_price[i])
                current_port_val = self.cash[i] + lpv + spv

                lpv_nex = long_portfolio_value(self.quantity['x'][i], nex_x_p)
                spv_nex = short_portfolio_value(self.quantity['y'][i], nex_y_p, self.short_side_init_price[i])
                
                if spv_nex <= 0:
                    # we loss all the money in the short side
                    # so need to exit the long side
                    self.port_val_minus_com[i] = (
                        self.cash[i] + lpv_nex - incur_commission(nex_x_p, self.quantity['x'][i])
                    )
                    
                    # forced to take position 0. this mean all the assets transformed into cash
                    self.position[i] = 0
                    self.quantity['y'][i] = 0
                    self.quantity['x'][i] = 0
                    self.cash[i] = self.port_val_minus_com[i]
                    self.port_val[i] = self.port_val_minus_com[i]
                else:
                    exit_commission = (incur_commission(nex_y_p, self.quantity['y'][i])
                                      +incur_commission(nex_x_p, self.quantity['x'][i]))
                    self.port_val[i] = self.cash[i] + lpv_nex + spv_nex
                    self.port_val_minus_com[i] = self.cash[i] + lpv_nex + spv_nex - exit_commission
                    self.position[i] = 2
                
                r[i] = self.port_val_minus_com[i] - current_port_val

        return r
    
    def update_state(self):
#         # concate next_input_history and next position to form next partial state
#         partial_state = tf.concat([self.input_history[self.t].T, position_num.one_hot(self.position, position_num)], 1)
        
#         # update state
#         self.state = self.state_encoding_model(partial_state)

        action_observation = tf.concat([
            tf.one_hot(self.position, rl_constants.position_num),
            tf.convert_to_tensor(self.input_history[self.t].T, dtype=tf.float32)
        ], 1)
    
        # use rnn to encode observationans and current stock state into next stock state
        stock_state = self.state_encoding_model(action_observation)
        
        # do normalization for total_portfolio_value
        # this is extremely important. if not normalized, the action will be highly biased.
        portfolio_state = np.array([
            self.port_val/rl_constants.initial_cash,
#             self.quantity['y'],
#             self.quantity['x']
        ]).T
        
        # stock state and portfolio state together form the whole environment state
        self.state = tf.concat([
            stock_state,
            portfolio_state
        ], 1)
#         self.state = stock_state
    
    def step(self, action):
        """Given the current state and action, return the reward, next state and done.
        This function should be called after reset.
        
        reward is of type numpy array. state is of type tensor. done is of type boolean.
        
        
        Arguments:
            action: a numpy array containing the current action for each training pair.

        Note that we follow the convention where the trajectory is indexed as s_0, a_0, r_0,
        s_1, ... . Therefore t is updated just after computing the reward is computed and
        before computing next state.
        """
        # r_t
        r = self.compute_reward(action) # also update the position for time t+1

        # t = t+1
        self.t += 1
        
        # compute s_(t+1)
        self.update_state()

        return r, self.state, (self.t+1) == self.trading_period