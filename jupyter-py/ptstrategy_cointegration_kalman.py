from ptstrategy import PTStrategy
from pykalman import KalmanFilter
import statsmodels.api as sm

import pandas as pd
import numpy as np
import datetime as dt
import backtrader as bt
import math

class CointKalmanStrategy(PTStrategy):
    def __init__(self):
        super().__init__()
        
        self.alpha = None
        self.intercept = None
        self.filtered_state_means = None
        self.filtered_state_covariances = None
        self.spread_std = None
        self.kf = None

    def update_enter_exit_levels(self):
        if (self.kf is None):
            Y = np.log(pd.Series(self.data0.get(size=self.max_lookback, ago=0)).values)[:, np.newaxis]
            X = np.log(pd.Series(self.data1.get(size=self.max_lookback, ago=0)).values)[:, np.newaxis]
            
            # observation matrix
            C = np.hstack((np.ones_like(X), X))
            C = C.reshape(self.max_lookback, 1, 2)

            # state transition matrix
            I = np.array([[[1, 0], 
                          [0, 1]]])
            T = I.repeat(self.max_lookback - 1, axis = 0)

            self.kf = KalmanFilter(em_vars=['transition_covariance', 
                                       'observation_covariance', 
                                       'initial_state_mean', 
                                       'initial_state_covariance'], 
                                   transition_matrices=T,
                                   observation_matrices=C,
                                   n_dim_state=2, 
                                   n_dim_obs=1)
            
            # run EM algorithm
            self.kf.em(X=Y, n_iter=10)
            self.spread_std = math.sqrt(self.kf.observation_covariance[0][0])
            
            # filtering
            filtered_state_means, filtered_state_covariances = self.kf.filter(X=Y)
            self.filtered_state_means, self.filtered_state_covariances = filtered_state_means[-1], filtered_state_covariances[-1]

            # update entry and exit levels
            self.upper_limit = self.enter_threshold_size * self.spread_std
            self.lower_limit = -1 * self.enter_threshold_size * self.spread_std
            self.up_medium = self.exit_threshold_size * self.spread_std
            self.low_medium = -1 * self.exit_threshold_size * self.spread_std
            
            self.alpha, self.intercept = self.filtered_state_means[1], self.filtered_state_means[0]
            self.allow_trade = False
 
        else:
            self.alpha, self.intercept = self.filtered_state_means[1], self.filtered_state_means[0]
            observation_t = np.array([math.log(self.data0[0])])
            observation_matrix_t = np.array([1, math.log(self.data1[0])]).reshape(1, 2)
            I = np.array([[[1, 0], 
                          [0, 1]]])

            self.filtered_state_means, self.filtered_state_covariances = (
                self.kf.filter_update(
                    self.filtered_state_means,
                    self.filtered_state_covariances,
                    observation=observation_t,
                    transition_matrix=I.reshape(2, 2),
                    observation_matrix=observation_matrix_t
                )
            )
            self.allow_trade = True

    def get_spread(self):
        return (math.log(self.data0[0]) - self.alpha * math.log(self.data1[0]) - self.intercept)

    def run_trade_logic(self):
        spread = self.get_spread()

        if self.status == 0:
            # "NO position" status
            # alpha must be > 0 to take position!!

            if spread > self.upper_limit:
                self.short_spread()
            elif spread < self.lower_limit:
                self.long_spread()

        elif self.status == 1:
            # "SHORT the spread" status
            # short data0, long data1
            if self.consider_borrow_cost: 
                PTStrategy.incur_borrow_cost(self.initial_price_data0, self.qty0)
            
            if spread < self.lower_limit:
                self.long_spread()
                
            elif spread < self.up_medium:
                self.exit_spread()
            
            else:
                long_pv = PTStrategy.long_portfolio_value(self.data1[0], self.qty1)
                short_pv = PTStrategy.short_portfolio_value(self.initial_price_data0, self.data0[0], self.qty0)
                net_gain_long = long_pv - self.initial_long_pv
                net_gain_short = short_pv - self.initial_short_pv

                return_of_current_trade = (net_gain_long + net_gain_short) / self.initial_cash

                # if losing too much money, exit
                if return_of_current_trade < self.loss_limit or short_pv <= 0:
                    self.exit_spread()

        elif self.status == 2:
            # "LONG the spread" status
            # short data1, long data0
            if self.consider_borrow_cost: 
                self.incur_borrow_cost(self.initial_price_data1, self.qty1)
            
            if spread > self.upper_limit:
                self.short_spread()
                
            elif spread > self.low_medium:
                self.exit_spread()
            
            else:
                long_pv = PTStrategy.long_portfolio_value(self.data0[0], self.qty0)
                short_pv = PTStrategy.short_portfolio_value(self.initial_price_data1, self.data1[0], self.qty1)
                net_gain_long = long_pv - self.initial_long_pv
                net_gain_short = short_pv - self.initial_short_pv

                return_of_current_trade = (net_gain_long + net_gain_short) / self.initial_cash

                # if losing too much money, exit
                if return_of_current_trade < self.loss_limit or short_pv <= 0:
                    self.exit_spread()
