python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_14_daily.csv" --save_result_name "3_28_daily/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_48_hourly.csv" --save_result_name "3_28_hourly/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_18_monthly.csv" --save_result_name "3_28_monthly/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_8_other.csv" --save_result_name "3_28_other/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_8_quarterly.csv" --save_result_name "3_28_quarterly/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_13_weekly.csv" --save_result_name "3_28_weekly/mase.csv"

python ./main.py --config-path "un_mase.json" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "univariate/uni_result_horizon_6_yearly.csv" --save_result_name "3_28_yearly/mase.csv"
