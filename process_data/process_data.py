import sys

sys.path.append("../model")
sys.path.append('./model')

import os
from os.path import isfile, isdir, join, splitext
import glob
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
import datetime

# custom
from process_raw_prices import get_filename_without_ext

plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["font.size"] = 12

def my_read_csv(p):
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def recreate_dir(folder):
    if folder == None:
        raise Exception("folder cannot be None.")
    elif len(folder) == 0:
        raise Exception("folder cannot be empty string.")
    elif False not in [c == "/" for c in list(folder)]:
        raise Exception("folder cannot be root.")

    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


def trim_raw_data_files(
    start_date=datetime.date(2015, 1, 1),
    end_date=datetime.date(2019, 1, 5),
    raw_folder="../../dataset/nyse-daily/",
    result_folder="../../dataset/nyse-daily-trimmed-same-length/",
    plot=False,
):

    raw_files_path_pattern = raw_folder + "*.csv"

    # raw dataset files pattern
    nyse_csv_paths = sorted(glob.glob(raw_files_path_pattern))
    lengths = [len(pd.read_csv(p)) for p in nyse_csv_paths]
    print("There are {} stock data csv files.".format(len(nyse_csv_paths)))
    if plot:
        plt.hist(lengths, 50)
        plt.xlabel("length")
        plt.ylabel("frequency")
        plt.title("Stock data length distribution")
        plt.gcf().set_size_inches(10, 5)
        plt.show()

    trimmed_df = {}

    for p in nyse_csv_paths:
        filename = get_filename_without_ext(p)
        df = my_read_csv(p)
        df = df[
            (pd.Timestamp(start_date) <= df["date"])
            & (df["date"] < pd.Timestamp(end_date))
        ]
        if len(df) > 0:
            trimmed_df[filename] = df

    lengths = [len(df) for fn, df in trimmed_df.items()]
    print("There are {} trimmed stock data.".format(len(trimmed_df)))
    if plot:
        plt.hist(lengths, 20)
        plt.xlabel("length")
        plt.ylabel("frequency")
        plt.title(
            "Length of stock data starting from {} distribution".format(str(start_date))
        )
        plt.gcf().set_size_inches(10, 5)
        plt.show()

    max_length = max(lengths)
    max_length_data = []
    for fn, df in trimmed_df.items():
        if len(df) == max_length:
            max_length_data.append((fn, df))

    # find intersection of the max length group of stocks
    intersection = max_length_data[0][1]["date"].values
    for fn, temp_df in max_length_data[1:]:
        intersection, _, __ = np.intersect1d(
            intersection,
            temp_df["date"].values,
            assume_unique=True,
            return_indices=True,
        )

    if len(intersection) == max_length:
        print(
            "All stock data starting from date {} with {} data points each have common trading date.".format(
                str(start_date), max_length
            )
        )

        # save those stocks in a result folder
        recreate_dir(result_folder)

        for fn, df in max_length_data:
            df.to_csv(path_or_buf=join(result_folder, fn + ".csv"), index=False)

        print("The processed dataset was placed in: {}".format(result_folder))
        print("There should be {} csv files.".format(len(max_length_data)))
        return dict(max_length_data)
    else:
        print(
            "All stock data starting from date {} with max length do not have common trading date.".format(
                str(start_date)
            )
        )
        raise Exception("ERROR: need to do some other complicated preprocessing")


if __name__ == "__main__":
    trim_raw_data_files()
