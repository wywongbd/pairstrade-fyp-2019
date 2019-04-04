import tensorflow as tf
import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import warnings
import glob
import os
import sys
import uuid
import itertools
import json
import argparse
import pathlib
import logging

from grid_search_tools import GSTools
from ptstrategy_cointegration_kalman import CointKalmanStrategy
from ptstrategy_distance import DistStrategy
from ptstrategy_cointegration import CointStrategy
from custom_analyzer import Metrics
from pandas_datafeed import PandasData
from datetime import datetime
from pytz import timezone
from decode_logs import get_current_time
from pair_selector import *

sys.path.append("../process-data")
sys.path.append("../log-helper")
from process_data import trim_raw_data_files
from log_helper import LogHelper
sys.path.pop()
sys.path.pop()

##################################################################################################
# Define parameters                                                                              #
##################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../ib-data/nyse-daily-tech/",
                    help="Path to stock price data")
parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory.")
parser.add_argument("--strategy_type", default="distance", type=str, choices=["distance", "cointegration", "kalman"],
                    help="Type of strategy used, either distance, cointegration, or kalman.")

parser.add_argument("--stk0", type=str, default="AAN",
                    help="Stock symbol of first stock.")
parser.add_argument("--stk1", type=str, default="AER",
                    help="Stock symbol of second stock.")

parser.add_argument("--kalman_estimation_length", type=int, default=50,
                    help="Number of days used for Kalman EM algorithm. Only useful if strategy is kalman.")

parser.add_argument("--backtest_start", type=str, default="2018-05-01",
                    help="Start date of backtest.")
parser.add_argument("--backtest_end", type=str, default="2018-12-31",
                    help="End date of backtest.")

parser.add_argument("--lookback", default=50, type=int, 
                    help="Lookback value to be tested. Only useful if strategy is distance or cointegration.")
parser.add_argument("--enter_threshold", default=2.0, type=float, 
                    help="Enter threshold value to be tested (in units 'number of SD from mean').")
parser.add_argument("--exit_threshold", default=0.5, type=float, 
                    help="Exit threshold value to be tested (in units 'number of SD from mean').")
parser.add_argument("--loss_limit", default=-0.005, type=float, 
                    help="Position will exit if loss exceeded this loss limit.")

config = parser.parse_args()

##################################################################################################
# Main function                                                                                  #
##################################################################################################

def main():
    ##################################################################################################
    # Setup logger and output dir                                                                    #
    ##################################################################################################
    output_dir = config.output_dir

    if output_dir is None:
        output_dir = 'output/backtest-{}'.format(get_current_time())
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    LogHelper.setup(log_path='{}/backtesting.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name__)

    # Log all paremeters
    _logger.info("Backtest parameters: {}".format(vars(config)))
    
    # load data
    data = GSTools.load_csv_files(config.data_path)
    stk0, stk1 = config.stk0, config.stk1
    
    # check existence of stocks
    if (stk0 not in data):
        _logger.error("Stock symbol {} does not exist!".format(stk0))
        return
    if (stk1 not in data):
        _logger.error("Stock symbol {} does not exist!".format(stk1))
        return
    
    # size requirements
    pre_backtest_size = None
    if config.strategy_type == "cointegration" or config.strategy_type == "distance":
        pre_backtest_size = config.lookback
    elif config.strategy_type == "kalman":
        pre_backtest_size = config.kalman_estimation_length
    
    # select segment of data that we want
    data0, data1 = data[stk0].set_index("date"), data[stk1].set_index("date")
    start_date_dt = datetime.strptime(config.backtest_start, "%Y-%m-%d").date()
    end_date_dt = datetime.strptime(config.backtest_end, "%Y-%m-%d").date() 
    
    data0 = data0[ : start_date_dt].tail(pre_backtest_size).append(data0[start_date_dt : end_date_dt])
    data1 = data1[ : start_date_dt].tail(pre_backtest_size).append(data1[start_date_dt : end_date_dt])
    data0 = data0.reset_index()
    data1 = data1.reset_index()
    
    # initialize cerebro
    cerebro = bt.Cerebro()

    # Create data feeds
    data0 = bt.feeds.PandasData(dataname=data0, timeframe=(bt.TimeFrame.Days), datetime=0, open=1, close=4)
    data1 = bt.feeds.PandasData(dataname=data1, timeframe=(bt.TimeFrame.Days), datetime=0, open=1, close=4)

    # add data feeds to cerebro
    cerebro.adddata(data0)
    cerebro.adddata(data1)

    # Add the strategy
    if config.strategy_type == "distance":
        cerebro.addstrategy(DistStrategy, 
                            stk0_symbol=stk0,
                            stk1_symbol=stk1,
                            lookback=config.lookback,
                            max_lookback=pre_backtest_size,
                            enter_threshold_size=config.enter_threshold, 
                            exit_threshold_size=config.exit_threshold, 
                            loss_limit=config.loss_limit,
                            consider_borrow_cost=True,
                            consider_commission=False,
                            print_msg=True,
                            print_transaction=True
                           )
    elif config.strategy_type == "cointegration":
        cerebro.addstrategy(CointStrategy, 
                            stk0_symbol=stk0,
                            stk1_symbol=stk1,
                            lookback=config.lookback,
                            max_lookback=pre_backtest_size,
                            enter_threshold_size=config.enter_threshold, 
                            exit_threshold_size=config.exit_threshold, 
                            loss_limit=config.loss_limit,
                            consider_borrow_cost=True,
                            consider_commission=True,
                            print_msg=True,
                            print_transaction=True
                           )
    elif config.strategy_type == "kalman":
        cerebro.addstrategy(CointKalmanStrategy, 
                            stk0_symbol=stk0,
                            stk1_symbol=stk1,
                            max_lookback=pre_backtest_size,
                            enter_threshold_size=config.enter_threshold, 
                            exit_threshold_size=config.exit_threshold, 
                            loss_limit=config.loss_limit,
                            consider_borrow_cost=True,
                            consider_commission=True,
                            print_msg=True,
                            print_transaction=True
                           )
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
    cerebro.addanalyzer(Metrics, 
                        lookback=pre_backtest_size, 
                        _name='metrics')

    # Add the commission - only stocks like a for each operation
    cerebro.broker.setcash(1000000)

    # And run it
    strat = cerebro.run()
    
    # get MICRO metrics
    results_dict = {}
    results_dict["pair"] = stk0 + "-" + stk1
    results_dict["sharperatio"]= strat[0].analyzers.mysharpe.get_analysis()['sharperatio']
    results_dict["returnstd"] = strat[0].analyzers.metrics.returns_std()
    results_dict["avg_holding_period"] = strat[0].analyzers.metrics.avg_holding_period
    results_dict["n_trades"] = strat[0].analyzers.metrics.n_trades
    results_dict["startcash"] = cerebro.getbroker().startingcash
    results_dict["endcash"] = cerebro.getbroker().getvalue()
    results_dict["profit"] = (results_dict["endcash"] - results_dict["startcash"]) / results_dict["startcash"]
    _logger.info("[pair-performance]: {}".format(results_dict))
    
if __name__ == '__main__':
    main()
