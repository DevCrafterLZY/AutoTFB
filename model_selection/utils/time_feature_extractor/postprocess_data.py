import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common.constant import *
from data.time_feature_extractor.utils.init_settings import load_defaults


def apply_pca(data_group):
    n, time_steps, features = data_group.shape
    reshaped_data = data_group.transpose(1, 2, 0).reshape(time_steps * features, n)  # shape (48*320, n)
    pca = PCA(n_components=1)
    reduced_data = pca.fit_transform(reshaped_data).reshape(time_steps, features)  # shape (48, 320)
    return reduced_data


if __name__ == "__main__":
    default_config = load_defaults()
    exp_id = default_config["general"]["exp_id"]
    data = np.load(
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, f"data_{exp_id}.pkl"),
        allow_pickle=True
    )
    column_ranges = pd.read_csv(
        os.path.join(FEATURE_EXTRACTION_DATASETS_PATH, "preprocessed_datasets_all_meta_info.csv")
    )

    results = []

    for _, row in column_ranges.iterrows():
        start_idx, end_idx = row["start_column"], row["end_column"]
        group_data = data[start_idx:end_idx + 1]  # 提取当前组数据

        if group_data.shape[0] == 1:
            results.append(group_data[0])
        else:
            reduced_data = apply_pca(group_data)
            results.append(reduced_data)

    results = np.array(results)
    with open(
            os.path.join(PERFORMANCE_PREDICTION_DATASETS_PATH, f"postprocessed_data_all_{exp_id}.pkl"),
            "wb"
    ) as f:
        pickle.dump(results, f)
