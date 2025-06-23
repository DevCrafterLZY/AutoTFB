import argparse
import logging
import os.path
import pickle

import numpy as np

from common.constant import *
from data.time_feature_extractor.models import TS2Vec
from data.time_feature_extractor.train import preprocess_data
from data.time_feature_extractor.utils.init_settings import load_defaults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(model_path, data, output_path=None):
    # Load the trained model
    model = TS2Vec(input_dims=1)
    model.load(model_path)

    logger.info("Encoding data...")
    encoded_data = model.encode(
        data, causal=True, sliding_length=1, sliding_padding=2, batch_size=256
    )

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(encoded_data, f)

    logger.info(f"Features saved to {output_path}")
    return encoded_data


def main(args):
    dataset = np.load(
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "preprocessed_datasets_all.npy"))
    data = preprocess_data(dataset)

    # Extract features
    extract_features(
        os.path.join(FEATURE_EXTRACTION_CHECKPOINT_PATH, f"model_{args.exp_id}.pkl"),
        data,
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, f"data_{args.exp_id}.pkl"),
    )


if __name__ == "__main__":
    # Load default values from the configuration file
    default_config = load_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_num",
        type=int,
        default=default_config["data"]["sample_num"],  # Default value from config file
        help=f"Number of samples (default: {default_config['data']['sample_num']})",
    )
    parser.add_argument(
        "--max-pred-len",
        type=int,
        default=default_config["data"]["max_pred_len"],  # Default value from config file
        help=f"Maximum prediction length (default: {default_config['data']['max_pred_len']})",
    )
    parser.add_argument(
        "--exp_id",
        type=int,
        default=default_config["general"]["exp_id"],  # Default value from config file
        help=f"Experiment ID (default: {default_config['general']['exp_id']})",
    )
    args = parser.parse_args()

    main(args)
