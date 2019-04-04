import math
import itertools
import logging

import pandas as pd
import numpy as np
import datetime as dt
import backtrader as bt

from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)

class PTStrategy(bt.Strategy):
    # abstract base class 

    #######################################################################################
    # parameters
    params = dict (
        lookback = 20,
        max_lookback = 30,
        enter_threshold_size = 2,
        exit_threshold_size = 0.5,
        loss_limit = -0.015,
        consider_borrow_cost = False,
        consider_commission = False,
        print_bar = True,
        print_msg = False,
        print_transaction = False,
        stk0_symbol = '',
        stk1_symbol = '',
    )

    #######################################################################################
    def __init__(self):
        # keeps track whether order is pending
        self.orderid = None

        # general info
        self.stk0_symbol = self.p.stk0_symbol
        self.stk1_symbol = self.p.stk1_symbol

        # Strategy params
        self.lookback = self.p.lookback
        self.max_lookback = self.p.max_lookback
        self.enter_threshold_size = self.p.enter_threshold_size
        self.exit_threshold_size = self.p.exit_threshold_size
        self.loss_limit = self.p.loss_limit
        self.consider_borrow_cost = self.p.consider_borrow_cost
        self.consider_commission = self.p.consider_commission

        # Parameters for printing
        self.print_bar = self.p.print_bar
        self.print_msg = self.p.print_msg
        self.print_transaction = self.p.print_transaction

        # temporary variables for trading logic
        self.status = 0
        self.qty0 = 0
        self.qty1 = 0
        self.initial_price_data0 = None
        self.initial_price_data1 = None
        self.initial_cash = None
        self.initial_long_pv = None
        self.initial_short_pv = None
        self.upper_limit = None
        self.lower_limit = None
        self.up_medium = None
        self.low_medium = None
        self.allow_trade = True
        
        # for logging
        self.latest_trade_action = None
        self.sell_stk = None
        self.buy_stk = None
        self.sell_amt = None
        self.buy_amt = None

    #######################################################################################
    @staticmethod
    def log(txt, dt=None, data=None):
        dt = dt or data.datetime[0]
        dt = bt.num2date(dt)
        _logger.info('%s, %s' % (dt.isoformat(), txt))

    @staticmethod
    def long_portfolio_value(price, qty):
        return price * qty

    @staticmethod
    def short_portfolio_value(price_initial, price_final, qty):
        return qty * (1.5 * price_initial - price_final)

    #######################################################################################
    @abstractmethod
    def log_status(self):
        pass

    @abstractmethod
    def update_enter_exit_levels(self):
        pass

    @abstractmethod
    def get_spread(self):
        pass

    @abstractmethod
    def run_trade_logic(self):
        pass

	#######################################################################################
    def next(self):
        # reset variable
        self.latest_trade_action = None
        self.sell_stk = None
        self.buy_stk = None
        self.sell_amt = None
        self.buy_amt = None

        # this is important for grid search, to ensure all trading strats start together
        if min(len(self.data0), len(self.data1)) <= self.max_lookback:
            return

        ###################################################################################
        # COMPUTE UPPER AND LOWER LIMITS                                                  #
        ###################################################################################
        if self.status == 0:
            self.update_enter_exit_levels()

        ###################################################################################
        # STRATEGY LOGIC                                                                  #
        ###################################################################################
        # if an order is active, no new orders are allowed
        if self.allow_trade and (not self.orderid):
            self.run_trade_logic()

        if self.print_msg:
            self.log_status()

    def short_spread(self):
        x = int((2 * self.broker.getvalue() / 3.0) / (self.data0.close[0]))  
        y = int((2 * self.broker.getvalue() / 3.0) / (self.data1.close[0]))  

        # Placing the order
        self.sell(data=self.data0, size=(x + self.qty0))  # Place an order for buying y + qty2 shares
        self.buy(data=self.data1, size=(y + self.qty1))  # Place an order for selling x + qty1 shares

        # Updating the counters with new value
        self.qty0 = x  
        self.qty1 = y  

        # update flags
        self.status = 1

        # keep track of trade variables
        self.initial_cash = self.qty1 * self.data1[0] + 0.5 * self.qty0 * self.data0[0]
        self.initial_long_pv = PTStrategy.long_portfolio_value(self.qty1, self.data1[0])
        self.initial_short_pv = 0.5 * self.data0[0] * self.qty0
        self.initial_price_data0, self.initial_price_data1 = self.data0[0], self.data1[0]
        
        # logging
        self.latest_trade_action = "short_spread"
        self.sell_stk = self.stk0_symbol
        self.buy_stk = self.stk1_symbol
        self.sell_amt = x + self.qty0
        self.buy_amt = y + self.qty1
    
    def long_spread(self):
        # Calculating the number of shares for each stock
        x = int((2 * self.broker.getvalue() / 3.0) / (self.data0[0])) 
        y = int((2 * self.broker.getvalue() / 3.0) / (self.data1[0])) 

        # Place the order
        self.buy(data=self.data0, size=(x + self.qty0))  # Place an order for buying x + qty1 shares
        self.sell(data=self.data1, size=(y + self.qty1))  # Place an order for selling y + qty2 shares

        # Updating the counters with new value
        self.qty0 = x 
        self.qty1 = y 

        # update flags
        self.status = 2  

        # keep track of trade variables
        self.initial_cash = self.qty0 * self.data0[0] + 0.5 * self.qty1 * self.data1[0]
        self.initial_long_pv = PTStrategy.long_portfolio_value(self.qty0, self.data0[0])
        self.initial_short_pv = 0.5 * self.data1[0] * self.qty1
        self.initial_price_data0, self.initial_price_data1 = self.data0[0], self.data1[0]

        # logging
        self.latest_trade_action = "long_spread"
        self.sell_stk = self.stk1_symbol
        self.buy_stk = self.stk0_symbol
        self.sell_amt = y + self.qty1
        self.buy_amt = x + self.qty0
    
    def exit_spread(self):
        # Exit position
        self.close(self.data0)
        self.close(self.data1)

        # logging
        self.latest_trade_action = "exit_spread"
        self.sell_stk = None
        self.buy_stk = None
        self.sell_amt = None
        self.buy_amt = None

        # update counters
        self.qty0 = 0
        self.qty1 = 0

        # update flags
        self.status = 0
        self.initial_cash = None
        self.initial_long_pv, self.initial_short_pv = None, None
        self.initial_price_data0, self.initial_price_data1 = None, None

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f' % order.executed.price
                
                if self.print_transaction:
                    PTStrategy.log(buytxt, order.executed.dt)
                
                if self.consider_commission:
                    self.incur_commission(order.executed.price, order.executed.size)
                    
            else:
                selltxt = 'SELL COMPLETE, %.2f' % order.executed.price
                
                if self.print_transaction:
                    PTStrategy.log(selltxt, order.executed.dt)
                    
                if self.consider_commission:
                    self.incur_commission(order.executed.price, order.executed.size)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            if self.print_transaction:
                PTStrategy.log('%s ,' % order.Status[order.status])

        # Allow new orders
        self.orderid = None

    def incur_commission(self, price, qty):
        qty = abs(qty)
        commission = min(max(1, 0.005*qty), 0.01*price*qty)
        self.broker.add_cash(-1*commission)

    def incur_borrow_cost(self, price, qty):
        # 0.25 percent (annualized) borrow cost
        cost = 0.0025 * abs(qty) * price / 365.0
        self.broker.add_cash(-1*cost) 

    def stop(self):
        if self.print_bar:
        	_logger.info("-")

        if self.print_msg:
        	_logger.info('==================================================')
        	_logger.info('Starting Value: %.2f' % self.broker.startingcash)
        	_logger.info('Ending   Value: %.2f' % self.broker.getvalue())
        	_logger.info('==================================================')

