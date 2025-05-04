import argparse
import ast
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from AutoTFB.NCF import NCF
from common.constant import (CONFIG_PATH, LIGHTGBM_MODEL_CHECKPOINT_PATH,
                             FEATURE_EXTRACTION_DATASETS_PATH,
                             PERFORMANCE_PREDICTION_DATASETS_PATH,
                             NCF_MODEL_CHECKPOINT_PATH, RESULT_PATH)
from utils import get_kfold_train_test_data, preprocess_data, train_NCF, train_test_gbm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    parser.add_argument("--config-path", default="un_rmse.json", type=str, help="Evaluation config file path", )
    parser.add_argument("--fold", default=5, type=int, help="K fold", )

    # model arguments
    parser.add_argument("--feature_dim1", type=int, default=320, help="Dimension of the first feature")
    parser.add_argument("--feature_dim2", type=int, default=24

                        , help="Dimension of the second feature")
    parser.add_argument("--hidden_dims", type=str, default="[8192, 4096, 2048]", help="Dimension of the hidden layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the model")
    parser.add_argument("--tsvec_dim", type=int, default=1024,
                        help="Dimension of k (typically in attention mechanisms)")
    parser.add_argument("--model_dim", type=int, default=1024,
                        help="Dimension of k (typically in attention mechanisms)")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--horizon", type=int, default=None, help="Which horizon of results should be selected.")

    # lightgbm arguments
    parser.add_argument("--gbm_min_data_in_leaf", type=int, default=50, help="Random seed for reproducibility")
    parser.add_argument("--gbm_lr", type=float, default=0.001, help="Patience for early stopping")
    parser.add_argument("--gbm_num_leaves", type=int, default=511, help="Which horizon of results should be selected.")

    # llm arguments
    parser.add_argument("--llm", type=str, default="Meta-Llama-3-8B", help="Llm name")
    parser.add_argument("--llm_repeat_time", type=int, default=3, help="Llm repeat time")
    parser.add_argument("--topK", type=int, default=3, help="The number of top models for select")
    parser.add_argument("--sort_metric", type=str, default="mae_norm", help="The metric for sort")

    # static file names
    parser.add_argument("--dataset_feature_file", type=str, default="un_feature.pkl",
                        help="Dataset feature file name")
    parser.add_argument("--all_results_file", type=str, default="un_hourly_horizon_48.csv",
                        help="All results file name")
    parser.add_argument("--dataset_meta_info_file", type=str, default="dataset_meta_info.csv",
                        help="Dataset meta info file name")
    parser.add_argument("--model_meta_info_file", type=str, default="model_meta_info.json",
                        help="Model meta info file name")
    parser.add_argument("--ncf_checkpoint_dir", type=str, default=None, help="NCF checkpoint path")
    parser.add_argument("--ncf_save_dir", type=str, default=None, help="NCF save path")
    parser.add_argument("--gbm_save_dir", type=str, default=None, help="gbm save path")
    # save file names
    parser.add_argument("--save_result_name", type=str, default=None, help="Save result file name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = {
        "boosting_type": "gbdt",
        "objective": "lambdarank",
        "metric": "ndcg",
        "min_data_in_leaf": args.gbm_min_data_in_leaf,
        "learning_rate": args.gbm_lr,
        "num_leaves": args.gbm_num_leaves,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.4,
        "lambda_l2": 0.5,
        "min_gain_to_split": 0.2,
        "ndcg_at": [3, 5, 7],
    }

    dataset_feature_file_path = os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, args.dataset_feature_file)
    all_results_file_path = os.path.join(PERFORMANCE_PREDICTION_DATASETS_PATH, args.all_results_file)
    config_path = os.path.join(CONFIG_PATH, args.config_path)

    with open(config_path, "rb") as f:
        config_data = json.load(f)

    target_metrics = config_data["target_metrics"]

    with open(dataset_feature_file_path, "rb") as f:
        dataset_feature = pickle.load(f)
    all_results = pd.read_csv(all_results_file_path)
    all_results = preprocess_data(all_results, target_metrics)
    all_results["model_name"] = all_results["model_name"] + "_" + all_results["model_params"].astype(str)
    all_results = all_results.sort_values(by=['file_name', 'model_name'])
    all_model_names = all_results["model_name"].unique().tolist()

    dataset_num = len(all_results["file_name"].unique().tolist())
    model_names_to_index = {name: index for index, name in enumerate(all_model_names)}

    args.output_dim = len(target_metrics)
    args.model_num = len(all_model_names)
    print(args.model_num)
    args.hidden_dims = ast.literal_eval(args.hidden_dims)

    kfold_data = get_kfold_train_test_data(
        all_results=all_results,
        k_folds=args.fold,
    )
    all_predict = []

    ncf = None
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    for fold_index, (train, test) in enumerate(kfold_data):
        if args.ncf_checkpoint_dir is not None:
            ncf = NCF(args)
            ncf_path = os.path.join(NCF_MODEL_CHECKPOINT_PATH, args.ncf_checkpoint_dir, f"{fold_index}.pth")
            ncf.load_state_dict(torch.load(ncf_path, weights_only=True))
            logger.info(f"Load NCF model from {ncf_path}")
        else:
            ncf = train_NCF(
                config=args,
                model_names_to_index=model_names_to_index,
                dataset_feature=dataset_feature,
                train=train,
                target_metric=target_metrics,
            )

            ncf_save_dir = args.ncf_save_dir if args.ncf_save_dir else timestamp
            NCF_params_path = os.path.join(
                NCF_MODEL_CHECKPOINT_PATH,
                f"{ncf_save_dir}/{fold_index}.pth"
            )
            os.makedirs(os.path.dirname(NCF_params_path), exist_ok=True)
            torch.save(ncf.state_dict(), NCF_params_path)

        ncf.eval()

        train = train.sort_values(by=['file_name', 'model_name'])
        test = test.sort_values(by=['file_name', 'model_name'])

        gbm_save_dir = args.gbm_save_dir if args.gbm_save_dir else timestamp
        gbm_params_path = os.path.join(
            LIGHTGBM_MODEL_CHECKPOINT_PATH,
            f"{gbm_save_dir}/{fold_index}.txt"
        )
        pred = train_test_gbm(
            ncf=ncf,
            params=params,
            dataset_feature=dataset_feature,
            train=train,
            test=test,
            target_metrics=target_metrics,
            model_names_to_index=model_names_to_index,
            gbm_params_path=gbm_params_path,
        )

        predict_target_metric_name = [f"predict_{metric}" for metric in target_metrics]
        test[predict_target_metric_name] = pred.reshape(-1, 1)
        all_predict.append(test)

    all_predict = pd.concat(all_predict)

    if args.save_result_name is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_result_name = f"{timestamp}.csv"
    else:
        save_result_name = args.save_result_name

    result_path = os.path.join(RESULT_PATH, save_result_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    all_predict.to_csv(result_path, index=False)
