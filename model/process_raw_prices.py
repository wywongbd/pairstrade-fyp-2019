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


def compute_intercept_and_pvalue(p1, p2):
    # log Y = intercept + log X + c
    Y, X = pd.Series(np.log(p1)), pd.Series(np.log(p2))

    # returns (intercept, pvalue)
    return (np.mean(Y - X), smts.adfuller(Y - X)[1])


def compute_alpha_beta(Y, X):
    ln_Y = np.log(Y)
    ln_X = np.log(X)
    _X = sm.add_constant(ln_X)

    model = sm.OLS(ln_Y, _X)
    results = model.fit()
    beta, alpha = results.params
    return alpha, beta


def generate_pair_df(df1, df2, training_period=52):
    df1_train, df1_test = df1[:training_period], df1[training_period:]
    df2_train, df2_test = df2[:training_period], df2[training_period:]
    testing_period = len(df1_test)

    intercept, pvalue = compute_intercept_and_pvalue(df1_train['close'], df2_train['close'])

    df_combined = pd.DataFrame()

    # date
    df_combined["date"] = df1_test["date"]

    # # datetime features
    # df_combined['year'] = df1_test['year']
    # df_combined['monthOfYear'] = df1_test['monthOfYear']
    # df_combined['dayOfMonth'] = df1_test['dayOfMonth']
    # df_combined['hourOfDay'] = df1_test['hourOfDay']
    # df_combined['minuteOfHour'] = df1_test['minuteOfHour']
    # df_combined['dayOfWeek'] = df1_test['dayOfWeek']
    # df_combined['dayOfYear'] = df1_test['dayOfYear']
    # df_combined['weekOfYear'] = df1_test['weekOfYear']
    # df_combined['isHoliday'] = df1_test['isHoliday']
    # df_combined['prevDayIsHoliday'] = df1_test['prevDayIsHoliday']
    # df_combined['nextDayIsHoliday'] = df1_test['nextDayIsHoliday']

    # spread and pvalue
    df_combined["spread"] = pd.Series(np.log(df1_test['close']) - np.log(df2_test['close']) - intercept)
    df_combined["zscore"] = (df_combined["spread"] - np.mean(df_combined["spread"])) / np.std(df_combined["spread"])
    df_combined["pvalue"] = pvalue

    # price information of both stocks
    df_combined["open1"] = df1_test["open"]
    df_combined["high1"] = df1_test["high"]
    df_combined["low1"] = df1_test["low"]
    df_combined["close1"] = df1_test["close"]
    df_combined["logClose1"] = np.log(df1_test["close"])

    df_combined["open2"] = df2_test["open"]
    df_combined["high2"] = df2_test["high"]
    df_combined["low2"] = df2_test["low"]
    df_combined["close2"] = df2_test["close"]
    df_combined["logClose2"] = np.log(df2_test["close"])

    alpha_t, beta_t = [], []
    for i in range(testing_period):
        current_t = training_period + i
        df1_window = df1[current_t-training_period:current_t]['close']
        df2_window = df1[current_t-training_period:current_t]['close']
        a, b = compute_alpha_beta(df1_window, df2_window)
        alpha_t.append(a)
        beta_t.append(b)

    df_combined["alpha"] = pd.Series(alpha_t)
    df_combined["beta"] = pd.Series(beta_t)

    return df_combined


def generate_pairs_training_data(raw_files_path_pattern="../../dataset/nyse-daily/*.csv",
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
    print("Collected %d stocks with the same length of history." % N_STOCKS_TAKEN)

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
