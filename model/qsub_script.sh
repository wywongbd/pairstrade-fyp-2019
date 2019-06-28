#PBS -N train_0_test_1
cd /home/u24027/fyp_public_repo/statistical-arbitrage-18-19
python model/rl_train.py --job_name train_0_test_1 --run_mode "train" --train_indices 0 --test_indices 1
