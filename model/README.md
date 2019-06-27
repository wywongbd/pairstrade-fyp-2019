# Train and Test RL Trading Agent
Note that the following steps can be done by running the script "rl_train.py". However, you may want to modify the code a bit to do what you want. I recommend you read through the following explanation before running the code.

The ipynb are outdated. We will use python scripts to do the experiements. The main script is "rl_train.py".

## Data Preparation and Transformation
Before we start the training and testing, we need to prepare data. Data are pandas dataframes in CSV file format containing price series of financial instruments. We assume all dataframes having at least two columns: "date" and "close". All dataframes should have at least 252\*4 rows and the first 252\*4 rows in them should have the same dates. (Otherwise, modify the setting in "rl_load_data.py" accordingly)

To see the default data path or modify the default one, please refer to line 629 to 641 of "rl_train.py". The `rl_load_data.load_data` function will grab CSV files from `raw_files_path_pattern`, do some transformation, and then save the result into `dataset_folder_path`. You can safely ignore the argument `filter_pairs` and set it to `None` if you want to use up all the data that match `raw_files_path_pattern`.

Note: The data transformation is slow since now the running process is only single-threaded. Also, the data transformation will only be triggered if the folder `dataset_folder_path` is empty. The result of the data transformation are pandas dataframes in CSV file format storing data of pairs of financial instruments in the following filename pattern "\<asset_name_1\>-\<asset_name_2\>-\<index\>.csv". The default code will split the 252\*4 rows into 4 252 dates. That's why we have the "\<index\>" being 0, 1, 2, 3. 

## Training
To keep track of different jobs, please specify the flag `--job_name <some job name>` to differentiate different jobs. Later you can find the log file and saved models of those jobs under the directory "model/logging/\<job name\>". To train models, please specify the flag `--run_mode "train"` when running "rl_train.py". Since we have divided the raw data into 4 different periods, we can specify the flag `--train_indices` and `--test_indices`.  You may also need to specify `--load_which_data`.

After training, plots will also be generated and saved under "model/logging/\<job name\>/plots". The png file name should explain itself.

## Example Code for Running the scripts
Assuming data are prepared as pandas dataframe in CSV file format and saved under the directory "model/dataset".
1. cd to the project repo
2. run `python model/rl_train.py --job_name train_0_1_test_2 --run_mode "train" --train_indices 0 1 --test_indices 2`

## Note
For scripts, we current only have with comission only. You may be interested in how the return is computed. For that, please refer to the class TradingEnvironment in "trading_env.py".
