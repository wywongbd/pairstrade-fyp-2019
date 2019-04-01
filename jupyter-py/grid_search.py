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
parser.add_argument("--strategy_type", default="distance", type=str, choices=["distance", "cointegration", "kalman"],
                    help="Type of strategy used, either distance, cointegration, or kalman")

parser.add_argument("--start_date", type=str, default="2018-01-01",
                    help="Stock data which starts <= start_date and ends >= end_date will be selected for backtesting.")
parser.add_argument("--end_date", type=str, default="2018-12-31",
                    help="Stock data which starts <= start_date and ends >= end_date will be selected for backtesting.")

parser.add_argument("--pair_selection_start_date", type=str, default="2018-01-02",
                    help="Start date of pair selection.")
parser.add_argument("--pair_selection_end_date", type=str, default="2018-01-31",
                    help="End date of pair selection.")

parser.add_argument("--param_estimation_start", type=str, default="2018-02-01",
                    help="Start date of parameter estimation. Only useful if strategy is kalman.")
parser.add_argument("--param_estimation_end", type=str, default="2018-02-31",
                    help="End date of parameter estimation. Only useful if strategy is kalman.")

parser.add_argument("--backtest_start", type=str, default="2018-03-01",
                    help="Start date of backtest.")
parser.add_argument("--backtest_end", type=str, default="2018-12-31",
                    help="End date of backtest.")

parser.add_argument("--pct", type=float, default=0.95, 
                    help="Top pct percentage of the pairs with lowest distance/lowest pvalue will be backtested.")
parser.add_argument("--lookback_values", default=[20, 30, 40, 50], nargs='+', type=int, 
                    help="Lookback values to be tested. Only useful if strategy is distance or cointegration.")
parser.add_argument("--enter_thresholds", default=[1.0, 1.5, 2.0], nargs='+', type=float, 
                    help="Enter threshold values to be tested (in units 'number of SD from mean').")
parser.add_argument("--exit_thresholds", default=[0.25, 0.5], nargs='+', type=float, 
                    help="Exit threshold values to be tested (in units 'number of SD from mean').")
parser.add_argument("--loss_limits", default=[-0.005], nargs='+', type=float, 
                    help="Position will exit if loss exceeded this loss limit.")

config = parser.parse_args()

def main():
    ##################################################################################################
    # Setup logger and output dir                                                                    #
    ##################################################################################################
    output_dir = 'output/test{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    LogHelper.setup(log_path='{}/backtesting.log'.format(output_dir), log_level=logging.INFO)
    _logger = logging.getLogger(__name__)

    # Log all paremeters
    _logger.info("Grid search parameters: {}".format(vars(config)))
    
    # get relevant stock data
    start_date_dt = datetime.strptime(config.start_date, "%Y-%m-%d").date()
    end_date_dt = datetime.strptime(config.end_date, "%Y-%m-%d").date()
    
    print(start_date_dt, end_date_dt)
    
    data = trim_raw_data_files(start_date=start_date_dt,
                               end_date=end_date_dt,
                               raw_folder="../ib-data/nyse-daily-tech/",
                               result_folder="../tmp-data/")
    
    # get aggregated close prices
    close_df = GSTools.get_aggregated_with_dates(data, col='close').set_index("date")
    
    ##################################################################################################
    # perform pair selection                                                                         #
    ##################################################################################################
    ps_start_dt = config.pair_selection_start_date
    ps_end_dt = config.pair_selection_end_date
    ps_df = close_df.loc[ps_start_dt : ps_end_dt]
    good_pairs = None
    param_combinations = None
    
    # total number of stocks remaining
    N = len(data.keys())

    # number of pairs of interest
    K = int(config.pct * N * (N-1) / 2)
    
    if config.strategy_type == "distance":
        good_pairs = select_pairs_for_all_combin(train_df=ps_df, test_df=None, 
                                                 config={'n': K, 
                                                         'score_function': distance_score, 
                                                         'series_transform': distance_transform}, 
                                                 plot=False)
    
    elif config.strategy_type == "cointegration" or config.strategy_type == "kalman":
        good_pairs = coint(df=ps_df, intercept=True, sig_level=0.005)
        good_pairs.sort(key=lambda x: x[2])
        good_pairs = good_pairs[0 : K]
    
    # log all selected pairs
    _logger.info("The selected pairs are: {}".format(good_pairs))
        
    ##################################################################################################
    # generate parameter space                                                                       #
    ##################################################################################################
    if config.strategy_type == "distance" or config.strategy_type == "cointegration":
        param_combinations = list(itertools.product(config.lookback_values, 
                                                    config.enter_thresholds, 
                                                    config.exit_thresholds, 
                                                    config.loss_limits))
        param_combinations = [dict(zip(["lookback", 
                                        "enter_threshold", 
                                        "exit_threshold", 
                                        "loss_limit"], values)) for values in param_combinations]
    elif config.strategy_type == "kalman":
        param_combinations = list(itertools.product(config.enter_thresholds, 
                                                    config.exit_thresholds, 
                                                    config.loss_limits))
        param_combinations = [dict(zip(["enter_threshold", 
                                        "exit_threshold", 
                                        "loss_limit"], values)) for values in param_combinations]
    ##################################################################################################
    # calculate max_lookback                                                                         #
    ##################################################################################################
    MAX_LOOKBACK = 0
    if config.strategy_type == "distance" or config.strategy_type == "cointegration":
        MAX_LOOKBACK = max(config.lookback_values)
    elif config.strategy_type == "kalman":
        MAX_LOOKBACK = len(close_df.loc[config.param_estimation_start : param_estimation_end])
    
    ##################################################################################################
    # perform grid search                                                                            #
    ##################################################################################################
    # list to store MACRO results
    macro_results = []
    
    for i, params in enumerate(param_combinations, 1):
        _logger.info("Running parameter combination " + str(i) + "/" + str(len(param_combinations)))
        _logger.info("Backtesting all pairs using parameters: {}".format(params))
        
        # list to store MICRO results
        results = []
        
        stock_data = close_df.loc[: config.backtest_start].tail(MAX_LOOKBACK)
        stock_data = stock_data.append(close_df.loc[config.backtest_start : config.backtest_end])
        
        for j, pair in enumerate(good_pairs):
            # get names of both stock
            _logger.info("Running pair " + str(j) + "/" + str(len(good_pairs)))
            _logger.info("Backtesting pair: {}".format(params))
            stk0, stk1 = pair

            # get data of both stock
            stk0_df_test = pd.DataFrame({'date':stock_data[stk0].index, 'close':stock_data[stk0].values}) 
            stk1_df_test = pd.DataFrame({'date':stock_data[stk1].index, 'close':stock_data[stk1].values}) 
            
            # Create a cerebro
            cerebro = bt.Cerebro()

            # Create data feeds
            data0 = bt.feeds.PandasData(dataname=stk0_df_test, timeframe=(bt.TimeFrame.Days), datetime=0)
            data1 = bt.feeds.PandasData(dataname=stk1_df_test, timeframe=(bt.TimeFrame.Days), datetime=0)

            # add data feeds to cerebro
            cerebro.adddata(data0)
            cerebro.adddata(data1)
            
            # Add the strategy
            if config.strategy_type == "distance":
                cerebro.addstrategy(DistStrategy, 
                                    lookback=params["lookback"],
                                    max_lookback=MAX_LOOKBACK,
                                    enter_threshold_size=params["enter_threshold"], 
                                    exit_threshold_size=params["exit_threshold"], 
                                    loss_limit=params["loss_limit"],
                                    consider_borrow_cost=True,
                                    consider_commission=False,
                                    print_msg=False
                                   )
            elif config.strategy_type == "cointegration":
                cerebro.addstrategy(CointStrategy, 
                                    lookback=params["lookback"],
                                    max_lookback=MAX_LOOKBACK,
                                    enter_threshold_size=params["enter_threshold"], 
                                    exit_threshold_size=params["exit_threshold"], 
                                    loss_limit=params["loss_limit"],
                                    consider_borrow_cost=True,
                                    consider_commission=False
                                   )
            elif config.strategy_type == "kalman":
                cerebro.addstrategy(CointKalmanStrategy, 
                                    max_lookback=MAX_LOOKBACK,
                                    enter_threshold_size=params["enter_threshold"], 
                                    exit_threshold_size=params["enter_threshold"], 
                                    loss_limit=params["loss_limit"],
                                    consider_borrow_cost=True,
                                    consider_commission=False
                                   )

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
            cerebro.addanalyzer(Metrics, 
                                lookback=MAX_LOOKBACK, 
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
            results_dict["startcash"] = cerebro.getbroker().startingcash
            results_dict["endcash"] = cerebro.getbroker().getvalue()
            results_dict["profit"] = (results_dict["endcash"] - results_dict["startcash"]) / results_dict["startcash"]

            results.append(results_dict)
            _logger.info("Performance of this pair: {}".format(results_dict))

        # convert to dataframe
        results_df = pd.DataFrame(results)

        # save as csv
        uuid_str = str(uuid.uuid4())
        path = output_dir + str(uuid_str) + ".csv" 
        results_df.to_csv(path_or_buf=path, index=False)

        # calculate MACRO attributes
        avg_sharpe_ratio = results_df['sharpe_ratio'].mean()
        median_sharpe_ratio = results_df['sharpe_ratio'].median()

        avg_overall_return = results_df['overall_return'].mean()
        median_overall_return = results_df['overall_return'].median()
        overall_return_std = results_df['overall_return'].std()
        
        if config.strategy_type == "distance" or config.strategy_type == "cointegration":
            tup = (params["lookback"], params["enter_threshold"], params["exit_threshold"], params["loss_limit"],
                   avg_sharpe_ratio,
                   median_sharpe_ratio,
                   avg_overall_return,
                   median_overall_return,
                   overall_return_std,
                   uuid_str)
                         
        elif config.strategy_type == "kalman":
            tup = (params["enter_threshold"], params["exit_threshold"], params["loss_limit"],
                   avg_sharpe_ratio,
                   median_sharpe_ratio,
                   avg_overall_return,
                   median_overall_return,
                   overall_return_std,
                   uuid_str)

        macro_results.append(tup)
        _logger.info("Performance of this set of parameters: {}".format(tup))

        # nextline
        _logger.info("")
    
    
if __name__ == '__main__':
    main()
