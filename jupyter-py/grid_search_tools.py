import math
import glob
import os
import uuid
import itertools

import pandas as pd
import numpy as np
import datetime as dt

class GSTools(object):

	@staticmethod
	def load_csv_files(dir_str):
		'''
		This function reads all csv from the given directory, stores them in a dictionary and returns it.

		- dir_str should be of the form "../ib-data/nyse-daily-tech/"
		- expected format: the csv files should have a 'date' column

		'''
		# read all paths
		csv_paths = sorted(glob.glob(dir_str + "*.csv"))

		# create python dictionary
		data = {}

		for path in csv_paths:
			# get the file names
			filename = os.path.basename(path)
			filename_without_ext = os.path.splitext(filename)[0]

			# read the csv file as dataframe
			df = pd.read_csv(path)
			df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
			data[filename_without_ext] = df 

		return data

	@staticmethod
	def get_trading_dates(data):
		'''
		This function returns all trading days available in the dataset. 
		Return type is pandas.Index

		'''
		dates = pd.Index([])

		for key in data.keys():
			# reset index 
			data[key] = data[key].reset_index(drop=True)

			dates = dates.union(pd.Index(data[key]['date']))

		return dates

	@staticmethod
	def sync_start_end(data):
		'''
		This function synchronizes the start and end date of all dataframes in a given dictionary. 
		Rows with dates not between MAX_START_DATE and MIN_END_DATE will be dropped.

		'''
		# get max starting date
		MAX_START_DATE = pd.Timestamp.min
		MIN_END_DATE = pd.Timestamp.max

		for key in data.keys():
			# reset index 
			data[key] = data[key].reset_index(drop=True)

			# max
			MAX_START_DATE = max(MAX_START_DATE, data[key]['date'].iloc[0])
			MIN_END_DATE = min(MIN_END_DATE, data[key]['date'].iloc[-1])

		# take subset of all dataframes
		for key in data.keys():
			mask = (data[key]['date'] >= MAX_START_DATE) & (data[key]['date'] <= MIN_END_DATE)
			data[key] = data[key].loc[mask]

			# reset index 
			data[key] = data[key].reset_index(drop=True)

		return (data, MAX_START_DATE, MIN_END_DATE)

	@staticmethod
	def cut_datafeeds(data, size):
		'''
		This function cuts all dataframes to the intended size, 
		drops all dataframes whose length is < size from the dictionary 
		'''
		del_ls = []
		for key in data.keys():
			# reset index, just in case
			data[key] = data[key].reset_index(drop=True)
			N = len(data[key])

			if N < size:
				del_ls.append(key)
			else:
				data[key] = data[key][N - size:]

				# reset index again
				data[key] = data[key].reset_index(drop=True)

		for key in del_ls:
			data.pop(key, None)

		return data

	@staticmethod
	def get_aggregated(data, col='close'):
		'''
		Returns a dataframe with all close prices aggregated together.

		'''
		agg_df = pd.DataFrame()

		for key in data.keys():
			agg_df[key] = data[key][col]

		return agg_df
    
	@staticmethod
	def get_aggregated_with_dates(data, col='close'):
		'''
		Returns a dataframe with all close prices aggregated together.

		'''
		agg_df = pd.DataFrame()

		for key in data.keys():
			agg_df[key] = data[key][col]
			agg_df["date"] = data[key]["date"]

		return agg_df