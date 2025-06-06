python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/VAR_model"

