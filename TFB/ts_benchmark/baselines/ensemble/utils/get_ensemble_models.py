from typing import List

from ts_benchmark.models import get_models, ModelFactory


def get_ensemble_models(all_model_configs: List) -> List[ModelFactory]:
    all_model_config = {}
    for model_config in all_model_configs:
        model_config.setdefault("adapter", None)
        model_config.setdefault("model_hyper_params", {})
    all_model_config["models"] = all_model_configs
    all_model_config["recommend_model_hyper_params"] = {"norm": True}

    return get_models(all_model_config)
