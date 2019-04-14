import os
from os.path import isfile, join, splitext
import time
import pandas as pd
import logging
import sys

sys.path.append("./model")

import rl_constants
import process_raw_prices

_logger = logging.getLogger(__name__)

df_columns = ['close1', 'close2', 'normalizedLogClose1', 'normalizedLogClose2', 'spread', 'alpha', 'beta', 'date']
col_name_to_ind = {}
for i, c in enumerate(df_columns):
    if c == 'close1':
        col_name_to_ind['y_close'] = i
    elif c == 'close2':
        col_name_to_ind['x_close'] = i
    elif c == 'spread':
        col_name_to_ind['spread'] = i
    elif c == 'date':
        col_name_to_ind['date'] = i


def load_data(dataset_folder_path='./model/dataset/nyse-daily-transformed',
              raw_files_path_pattern="./model/dataset/nyse-daily-trimmed-same-length/*.csv",
              filter_pairs=None
             ):
    _logger.info("raw csv files loaded from {}".format(raw_files_path_pattern))
    
    os.makedirs(dataset_folder_path, exist_ok=True)

    # compute dataset
    all_pairs_slices = [splitext(f)[0] for f in os.listdir(dataset_folder_path) if isfile(join(dataset_folder_path, f))]
    if len(all_pairs_slices) == 0:
        process_raw_prices.generate_pairs_training_data(
            raw_files_path_pattern=raw_files_path_pattern,
            result_path=dataset_folder_path,
            min_size=252*rl_constants.num_of_period,
            training_period=52,
            points_per_cut=252
        )

    #     generate_pairs_data(raw_files_path_pattern, result_path=dataset_folder_path)
        all_pairs_slices = [splitext(f)[0] for f in os.listdir(dataset_folder_path) if isfile(join(dataset_folder_path, f))]
    _logger.info("Total number of pair slices: {}".format(len(all_pairs_slices)))

    # split for training and testing
    if filter_pairs == None:
        all_pairs = sorted(list(set(['-'.join(p.split('-')[0:2]) for p in all_pairs_slices])))
        rl_constants.num_of_pair = len(all_pairs)
    else:
        all_pairs = []
        for p in filter_pairs:
            s1, s2 = p.split('-')
            all_pairs.append('-'.join([s2, s1]))
        all_possible_pairs = set(['-'.join(p.split('-')[0:2]) for p in all_pairs_slices])
        all_pairs.extend(filter_pairs)
        all_pairs = filter(lambda p: p in all_possible_pairs, all_pairs)

#     all_pairs = ["VMW-WUBA"]
#     all_pairs = ["MSI-PANW"]
#     all_pairs = ["AAN-AER"]
    all_pairs_slices = [[] for i in range(rl_constants.num_of_period)]
    for p in all_pairs:
        for i in range(rl_constants.num_of_period):
            all_pairs_slices[i].append(p+'-{}'.format(i))

    for i in range(rl_constants.num_of_period):
        _logger.info("Total number of pair slices for period {}: {}".format(i, len(all_pairs_slices[i])))

    start_t = time.time()
    all_pairs_df = {}
    for i in range(len(all_pairs_slices)):
        for s in all_pairs_slices[i]:
            all_pairs_df[s] = pd.read_csv(join(dataset_folder_path, s+".csv"))

    _logger.info('time spent for loading df = {}s'.format(time.time()-start_t))
    
    for df_name, df in all_pairs_df.items():
        trading_period = len(df)
        break
    
#     for i, lis in enumerate(all_pairs_slices):
#         print("i = {}".format(i))
#         print("start date = {}".format(all_pairs_df[lis[0]]["date"].iloc[0]))
#         print("end date = {}".format(all_pairs_df[lis[0]]["date"].iloc[-1]))
        
    _logger.info("trading_period is {}".format(trading_period))

    # num_in_epoch = len(all_pairs_slices[0])
    return all_pairs_slices, all_pairs_df, trading_period