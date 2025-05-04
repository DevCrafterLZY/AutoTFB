import logging

import numpy as np
import pandas as pd

from common.constant import *
from data.time_feature_extractor.utils import process_data_df, load_defaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    default_config = load_defaults()
    sample_num = default_config["data"]["sample_num"]
    max_pred_len = default_config["data"]["max_pred_len"]
    datasets_train_folder_path = os.path.join(
        FEATURE_EXTRACTION_DATASETS_PATH,
        default_config["data"]["datasets_train_path"]
    )

    all_datasets = []
    column_ranges = []
    current_column_start = 0

    for filename in os.listdir(datasets_train_folder_path):
        file_path = os.path.join(datasets_train_folder_path, filename)

        if os.path.isfile(file_path) and file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
            df = process_data_df(data)
            subset = df[-(sample_num + max_pred_len): -max_pred_len]

            if subset.shape[0] == sample_num:
                all_datasets.append(subset.values)

                num_columns = subset.shape[1]
                column_ranges.append({
                    "filename": filename,
                    "start_column": current_column_start,
                    "end_column": current_column_start + num_columns - 1
                })
                current_column_start += num_columns
            else:
                logger.warning(f"{filename} skipped (insufficient rows).")

    if all_datasets:
        result = np.hstack(all_datasets)
        logger.info(f"Final shape: {result.shape}")
        np.save(os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "preprocessed_datasets_all.npy"), result)
        column_ranges_df = pd.DataFrame(column_ranges)
        column_ranges_df.to_csv(
            os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "preprocessed_datasets_all_meta_info.csv"),
            index=False
        )
        logger.info("Column ranges saved to 'column_ranges.csv'.")
    else:
        logger.info("No valid datasets found.")
