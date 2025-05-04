# -*- coding: utf-8 -*-
import os

# Get the root path where the code file is located
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

# Dataset paths
DATASETS_PATH = os.path.join(ROOT_PATH, "dataset")

DATASETS_TEMP_PATH = os.path.join(DATASETS_PATH, "temp")

FEATURE_EXTRACTION_DATASETS_PATH = os.path.join(DATASETS_PATH, "dataset_feature")

PERFORMANCE_PREDICTION_DATASETS_PATH = os.path.join(DATASETS_PATH, "model_performance")

TIME_SERIES_DATASETS_PATH = os.path.join(DATASETS_PATH, "time_series")

# Static path
STATIC_PATH = os.path.join(ROOT_PATH, "static")

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")

FEATURE_EXTRACTION_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "feature_extractor")

LLM_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "llm")

MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, "model")

NCF_MODEL_CHECKPOINT_PATH = os.path.join(MODEL_CHECKPOINT_PATH, "NCF")

LIGHTGBM_MODEL_CHECKPOINT_PATH = os.path.join(MODEL_CHECKPOINT_PATH, "lightgbm")

# Result path
RESULT_PATH = os.path.join(ROOT_PATH, "result")

# Config path
CONFIG_PATH = os.path.join(ROOT_PATH, "config")
