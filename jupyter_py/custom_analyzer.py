import pandas as pd
import numpy as np
import datetime as dt

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind

class Metrics(bt.Analyzer):
    params = dict(
        lookback=10,
    )

    def __init__(self):
        super(Metrics, self).__init__()

        self.pv = []
        self.status = []
        self.returns = None

        ###############################################################
        #  Trade statistics                                           #
        ###############################################################

        # number of resolved trades
        self.n_trades = 0

        # a trade (long or short) is not resolved if the position hasn't changed until the end of backtest 
        self.n_resolved_trades = 0
        self.n_unresolved_trades = 0

        # average holding period of resolved trades, -1 if there aren't any
        self.avg_holding_period = -1

        # length of unresolved trade, -1 if there aren't any
        self.len_unresolved_trade = -1

    def start(self):
        pass

    def next(self):
        if min(len(self.strategy.data0), len(self.strategy.data1)) >= self.p.lookback:
            self.pv.append(self.strategy.broker.getvalue())
            self.status.append(self.strategy.status)
            
    def stop(self):
        # convert lists to series
        self.pv = pd.Series(self.pv)
        
        # calculate returns
        self.returns = self.pv.diff()[1:]
        
        # calculate number of trades
        self.compute_trade_statistics()
        
    def compute_trade_statistics(self):
        _n = 0
        _mean = 0
        _counter = 0
        _curstate = 0
        
        for i, status in enumerate(self.status):
            if _curstate == 0:
                if status == 0:
                    continue
                else:
                    # entered position
                    _curstate = status
                    _counter = 1
                    
            else:
                if status == 0 or status != _curstate:
                    # changed position
                    _mean = (_n * _mean + _counter) / float(_n + 1)
                    _n += 1
                    _counter = 1
                    _curstate = status
                    
                elif status == _curstate:
                    _counter += 1
        
        self.n_resolved_trades = _n 
        self.n_unresolved_trades = 0 if (_curstate == 0) else 1
        self.n_trades = self.n_resolved_trades + self.n_unresolved_trades
        self.avg_holding_period = _mean if (self.n_resolved_trades > 0) else -1
        self.len_unresolved_trade = _counter if (self.n_unresolved_trades == 1) else -1
        
    def portfolio_value(self):
        return self.pv

    def returns_std(self):
        return self.returns.std()