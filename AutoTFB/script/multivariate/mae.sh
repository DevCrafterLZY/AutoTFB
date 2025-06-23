python ./main.py --config-path "mult_mae.json" --fold 7 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_24.csv" --save_result_name "multivariate/mu_24_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 7 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_36.csv" --save_result_name "multivariate/mu_36_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 7 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_48.csv" --save_result_name "multivariate/mu_48_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 7 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_60.csv" --save_result_name "multivariate/mu_60_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 18 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_96.csv" --save_result_name "multivariate/mu_96_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 18 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_192.csv" --save_result_name "multivariate/mu_192_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 18 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_336.csv" --save_result_name "multivariate/mu_336_mae.csv"

python ./main.py --config-path "mult_mae.json" --fold 18 --dataset_feature_file "mult_feature.pkl" --hidden_dims "[4096, 2048]" --tsvec_dim 1024 --model_dim 1024 --gpu_id 0 --gbm_lr 0.00001 --gbm_num_leaves 255 --gbm_min_data_in_leaf 10 --all_results_file "multivariate/mult_results_horizon_720.csv" --save_result_name "multivariate/mu_720_mae.csv"
