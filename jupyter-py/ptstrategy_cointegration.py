from ptstrategy import PTStrategy
import statsmodels.api as sm

import pandas as pd
import numpy as np
import datetime as dt
import backtrader as bt
import math

class CointStrategy(PTStrategy):
    def __init__(self):
        super().__init__()

        self.resid_mean = None
        self.resid_std = None
        self.alpha = None
        self.intercept = None

    def update_enter_exit_levels(self):
        Y = np.log(pd.Series(self.data0.get(size=self.lookback, ago=0)))
        X = np.log(pd.Series(self.data1.get(size=self.lookback, ago=0)))
        _X = sm.add_constant(X)

        model = sm.OLS(Y, _X)
        results = model.fit()
        self.intercept, self.alpha = results.params

        self.spread_mean = (Y - self.alpha * X - self.intercept).mean()
        self.spread_std = (Y - self.alpha * X - self.intercept).std()

        self.upper_limit = self.spread_mean + self.enter_threshold_size * self.spread_std
        self.lower_limit = self.spread_mean - self.enter_threshold_size * self.spread_std
        self.up_medium = self.spread_mean + self.exit_threshold_size * self.spread_std
        self.low_medium = self.spread_mean - self.exit_threshold_size * self.spread_std

    def get_spread(self):
        return (math.log(self.data0[0]) - self.alpha * math.log(self.data1[0]) - self.intercept)

    def run_trade_logic(self):
        spread = self.get_spread()

        if self.status == 0:
            # "NO position" status
            # alpha must be > 0 to take position?

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
