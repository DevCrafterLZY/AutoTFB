import argparse
import datetime
import logging
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.constant import *
from data.time_feature_extractor.models import TS2Vec
from data.time_feature_extractor.utils import init_dl_program, fix_random_seed, load_defaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_checkpoint_callback(save_every=1, unit="epoch", run_dir=None):
    """Creates a callback function to save model checkpoints."""
    assert unit in ("epoch", "iter")

    def callback(model, loss):
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0 and run_dir is not None:
            model.save(f"{run_dir}/model_{n}.pkl")

    return callback


def load_dataset(dataset_algorithm, dataset_path, sample_num, max_pred_len):
    """Loads and preprocesses the dataset."""
    dataset_list = []
    for dataset, _ in dataset_algorithm:
        data = pd.read_csv(os.path.join(dataset_path, dataset))
        df = data[["data"]][-(sample_num + max_pred_len): -max_pred_len]
        dataset_list.append(df.values)
    return np.concatenate(dataset_list, axis=1)


def preprocess_data(dataset):
    """Preprocesses the data by scaling and reshaping."""
    scaler = StandardScaler().fit(dataset)
    scaled_data = scaler.transform(dataset)

    # Adjust data dimensions
    if scaled_data.ndim == 2:
        scaled_data = np.expand_dims(scaled_data, 0)
    elif scaled_data.ndim == 1:
        scaled_data = np.expand_dims(scaled_data, 0)
        scaled_data = np.expand_dims(scaled_data, -1)

    return np.transpose(scaled_data, (2, 1, 0))


def main(args):
    # Fix random seed for reproducibility
    fix_random_seed()
    logger.info("Arguments: %s", str(args))

    # Initialize the device
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    # Load dataset information
    dataset_algorithm = np.load(
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "dataset_algorithm.npy"), allow_pickle=True
    )
    dataset = load_dataset(
        dataset_algorithm,
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "datasets_train"),
        args.sample_num,
        args.max_pred_len
    )

    logger.info("Loading and preprocessing data... ")
    data = preprocess_data(dataset)
    logger.info("done")

    # Configure model
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
    )

    # Add checkpoint saving callback if required
    if args.save_every is not None:
        unit = "epoch" if args.epochs is not None else "iter"
        config[f"after_{unit}_callback"] = save_checkpoint_callback(
            args.save_every, unit, FEATURE_EXTRACTION_CHECKPOINT_PATH
        )

    # Set up the run directory
    run_dir = FEATURE_EXTRACTION_CHECKPOINT_PATH
    os.makedirs(run_dir, exist_ok=True)

    # Initialize and/or load the model
    model = TS2Vec(input_dims=1, device=device, **config)
    model_path = os.path.join(FEATURE_EXTRACTION_CHECKPOINT_PATH, f"model_{args.exp_id}.pkl")
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        logger.info("Training...")
        start_time = time.time()
        model.fit(data, n_epochs=args.epochs, n_iters=args.iters, verbose=True)
        model.save(model_path)
        elapsed_time = time.time() - start_time
        logger.info("Training time: %s", datetime.timedelta(seconds=elapsed_time))

    logger.info("Training complete.")


if __name__ == "__main__":
    default_config = load_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=default_config["general"]["gpu"],
        help=f"GPU ID for training (default: {default_config['general']['gpu']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config["model"]["batch_size"],
        help=f"Batch size (default: {default_config['model']['batch_size']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_config["model"]["lr"],
        help=f"Learning rate (default: {default_config['model']['lr']})",
    )
    parser.add_argument(
        "--repr_dims",
        type=int,
        default=default_config["model"]["repr_dims"],
        help=f"Representation dimensions (default: {default_config['model']['repr_dims']})",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=default_config["data"]["sample_num"],
        help=f"Number of samples (default: {default_config['data']['sample_num']})",
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=default_config["model"]["max_train_length"],
        help=f"Maximum training sequence length (default: {default_config['model']['max_train_length']})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=default_config["model"]["num_iters"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_config["model"]["num_epochs"],
        help=f"Number of epochs (default: {default_config['model']['num_epochs']})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=default_config["model"]["save_every"],
        help="Checkpoint save frequency",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_config["general"]["seed"],
        help=f"Random seed (default: {default_config['general']['seed']})",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=default_config["general"]["max_threads"],
        help="Maximum threads",
    )
    parser.add_argument(
        "--max-pred-len",
        type=int,
        default=default_config["data"]["max_pred_len"],
        help=f"Maximum prediction length (default: {default_config['data']['max_pred_len']})",
    )
    parser.add_argument(
        "--exp_id",
        type=int,
        default=default_config["general"]["exp_id"],
        help=f"Experiment ID (default: {default_config['general']['exp_id']})",
    )
    args = parser.parse_args()

    main(args)
