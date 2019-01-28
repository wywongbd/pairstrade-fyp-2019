import os
import glob

import pandas as pd
import numpy as np

import statsmodels.tsa.stattools as smts
import statsmodels.api as sm

import itertools


def get_filename_without_ext(path):
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


# def compute_intercept_and_pvalue(p1, p2):
#     # log Y = intercept + log X + c
#     Y, X = pd.Series(np.log(p1)), pd.Series(np.log(p2))

#     # returns (intercept, pvalue)
#     return (np.mean(Y - X), smts.adfuller(Y - X)[1])


def compute_alpha_beta(Y, X):
    _X = sm.add_constant(X)

    model = sm.OLS(Y, _X)
    results = model.fit()
    beta, alpha = results.params
    return alpha, beta


def normalize_array(arr, train_arr):
    return (arr - np.mean(arr))/np.std(train_arr)


# def normalize_log_close(df, training_period):
#     return normalize_array(df['logClose'].values, df[:training_period]['logClose'].values)


def split_train_test(df, training_period):
    return df[:training_period], df[training_period:]


def compute_spread(Y, X, alpha, beta):
    return Y - alpha*X - beta


def compute_rolling_data(arrs, training_period, data_num, func):
    # assume all arrs are of the same length
    testing_period = len(arrs[0]) - training_period
    # initialize empty stats
    stats = np.zeros((data_num, testing_period))
    # use rolling windows
    for i in range(testing_period):
        windows_with_curr_val = [arr[i:training_period+i+1] for arr in arrs]
        stats[:, i] = func(*windows_with_curr_val)
    
    if data_num == 1:
        return stats[0]
    else:
        return stats


def compute_alpha_beta_exclude_current(Y, X):
    return compute_alpha_beta(Y[:-1], X[:-1])


def compute_rolling_alpha_beta(df1, df2, training_period):
    return compute_rolling_data(
        [df1['normalizedLogClose'].values, df2['normalizedLogClose'].values],
        training_period, 2, compute_alpha_beta_exclude_current
    )


def compute_normalize_current(arr):
    return (arr[-1] - np.mean(arr[:-1]))/np.std(arr[:-1])


def compute_rolling_normalization(df, training_period):
    return compute_rolling_data(
        [df['logClose'].values],
        training_period, 1, compute_normalize_current
    )


def generate_pair_df(df1, df2, training_period=52):
    # assume df1 and df2 have the same length and dates
    
    # compute log price
    df1['logClose'] = np.log(df1['close'].values)
    df2['logClose'] = np.log(df2['close'].values)
    
    df1_train, df1_test = split_train_test(df1, training_period)
    df2_train, df2_test = split_train_test(df2, training_period)

    # the resulting df
    df_combined = pd.DataFrame()

    # date
    df_combined["date"] = df1_test["date"].values
    
    # raw prices
    df_combined['close1'] = df1_test["close"].values
    df_combined['close2'] = df2_test["close"].values

    # rolling normalized log price
    df_combined['normalizedLogClose1'] = compute_rolling_normalization(df1, training_period)
    df_combined['normalizedLogClose2'] = compute_rolling_normalization(df2, training_period)
    df1_train_log_close = df1_train['logClose'].values
    df2_train_log_close = df2_train['logClose'].values
    df1['normalizedLogClose'] = np.concatenate((normalize_array(df1_train_log_close, df1_train_log_close),
                                                df_combined['normalizedLogClose1']))
    df2['normalizedLogClose'] = np.concatenate((normalize_array(df2_train_log_close, df2_train_log_close),
                                                df_combined['normalizedLogClose2']))
    
    # rolling computed alpha and beta
    df_combined["alpha"], df_combined["beta"] = compute_rolling_alpha_beta(df1, df2, training_period)
    
    df_combined["spread"] = compute_spread(df_combined['normalizedLogClose1'].values,
                                           df_combined['normalizedLogClose2'].values,
                                           df_combined["alpha"].values,
                                           df_combined["beta"].values)
    
#     df_combined["zscore"] = normalize_array(df_combined["spread"].values, train_spread)

    return df_combined


def generate_pairs_training_data(raw_files_path_pattern,
                                 result_path="../../dataset/nyse-daily-transformed",
                                 points_per_cut=252, min_size=252*4, training_period=52):

    nyse_csv_paths = sorted(glob.glob(raw_files_path_pattern))
    print("Collected %d stocks in raw data." % len(nyse_csv_paths))

    data = {}
    N_STOCKS_TAKEN = 0

    for path in nyse_csv_paths:
        filename_without_ext = get_filename_without_ext(path)

        # read the csv file as dataframe
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # if price history is long enough, we take it
        if len(df) >= min_size:
            data[filename_without_ext] = df
            N_STOCKS_TAKEN += 1
    print("Collected %d stocks with at least %d data points." % (N_STOCKS_TAKEN, min_size))

    stocks = list(data.keys())
    TOTAL_NUM_OF_PAIRS = len(stocks) * (len(stocks)-1) // 2
    i = 0
    num_cut = min_size // points_per_cut
    for stk1, stk2 in itertools.combinations(stocks, 2):
        df1, df2 = data[stk1], data[stk2]

        for j in range(num_cut):
            df1_slice = df1[j*points_per_cut:(j+1)*points_per_cut]
            df2_slice = df2[j*points_per_cut:(j+1)*points_per_cut]

            df3 = generate_pair_df(df1_slice, df2_slice, training_period=training_period)

            PATH = stk1 + "-" + stk2 + "-" + str(j)
            df3.to_csv(path_or_buf=os.path.join(result_path, PATH+".csv"), index=False)

        # print message
        print(str(i + 1) + "/" + str(TOTAL_NUM_OF_PAIRS) + " completed.")
        i += 1
