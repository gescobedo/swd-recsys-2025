from src.utils.helper import (
    iter_configs,
    yaml_load,
    yaml_save,
    get_results_dir,
    save_experiment_config,
    reproducible,
    merge_dicts,
    flatten_config,
    get_devices,
)
import torch
import json

from src.utils.input_validation import parse_input, find_rec_models
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from typing import List, Dict
from src.config.module_loader import (
    get_model,
    get_optimizer,
    get_trainer,
    get_attacker,
    get_atk_trainer,
)
from src.data.data_loading import (
    get_dataset_and_dataloader,
    get_atk_dataset_and_dataloader,
    get_rec_dataset,
)
from src.utils.input_validation import find_rec_models
from src.data.entity_feature import FeatureDefinition
from copy import deepcopy


def get_wandb_conf(config: dict):
    print(config)
    base_config = yaml_load(config["base_config"])
    results_dir = setup_results_dir(base_config)
    base_config["results_dir"] = os.path.join(
        results_dir,
        str(config["dataset_config"]["fold"]),
        "train",
        str(flatten_config(config)),
    )
    base_config = merge_dicts(base_config, config)
    print("=" * 80)
    print("Final config is\n", json.dumps(base_config, indent=4))
    print("=" * 80)
    base_config["dataset_config"].update(
        {
            "user_features": [
                FeatureDefinition(**d)
                for d in base_config["dataset_config"]["user_features"]
            ]
        }
    )

    return base_config


def generate_job_configuration(
    input_config: dict, results_dirs: List[Dict[str, object]]
):
    attacker_config = yaml_load(input_config["atk_config"])
    params = []
    devices = input_config["devices"]
    n_parallel = input_config["n_parallel"]
    devices = get_devices(devices, n_parallel)
    max_jobs = len(devices)
    verbose = max_jobs == 1

    for rd in results_dirs:
        atk_results_dir = os.path.join(rd["results_dir"])
        conf = deepcopy(attacker_config)
        conf.update(
            {
                "rec_model_config": yaml_load(rd["model_config"]),
                "rec_model_dict": rd["model_dict"],
                "results_dir": atk_results_dir,
            }
        )
        conf["random_state"] = conf["rec_model_config"]["random_state"]
        # Setting up the embbedding size
        embedding_size = conf["rec_model_config"]["model_config"]["latent_size"]
        conf["model_config"].update({"embedding_size": embedding_size})
        conf["verbose"] = verbose
        conf["dataset_config"].update(conf["rec_model_config"]["dataset_config"])
        conf.update({"use_wandb": conf["rec_model_config"].get("use_wandb", False)})

        params.append(conf)

    # Setting up devices

    if len(params) > len(devices):
        devices *= len(params) // len(devices) + 1
    current_device = 0
    for param in params:
        param.update({"device": devices[current_device]})
        current_device += 1
    return params, max_jobs


def setup_results_dir(input_config: dict) -> List[Dict]:
    base_dir = input_config.get("experiment", None)
    model_pattern = input_config.get("model_pattern", "**/*train*/**/*best_model*.pt")
    model_pattern = "**/*train*/**/*best_model*.pt"
    results_dirs = find_rec_models(base_dir, model_pattern)
    # print(results_dirs)
    for r_dict in results_dirs:
        r_dict.update({"results_dir": r_dict["results_dir"].replace("train", "atk")})
    # Creating atk folders
    for rd in results_dirs:
        os.makedirs(rd["results_dir"], exist_ok=True)
    return results_dirs


def configure_experiment():
    input_config = parse_input(
        "experiments",
        options=[
            "gpus",
            "n_parallel",
            "atk_config",
            "experiment",
            "model_pattern",
        ],
        access_as_properties=False,
    )

    results_dirs = setup_results_dir(input_config)
    params, max_jobs = generate_job_configuration(input_config, results_dirs)
    return params, max_jobs


def run_atk_experiments(params: List[Dict], njobs: int, training_fn):
    print("Running", len(params), "training job(s)")
    results = Parallel(n_jobs=min(njobs, len(params)), verbose=11)(
        delayed(training_fn)(p)
        for p in tqdm(params, desc="Running experiments", position=0, leave=True)
    )
    print("Done.")
    return results


def run_atk_train_test(config: dict):
    reproducible(config["random_state"])
    splits = ["train", "val", "test"]
    config["dataset_config"]["splits"] = splits
    datasets, dataloaders = get_atk_dataset_and_dataloader(config)
    train_dataset = datasets[splits[0]]

    rec_model_conf = config.get("rec_model_config", None)
    rec_model_dict = config.get("rec_model_dict", None)
    rec_model = get_model(rec_model_conf, train_dataset)
    rec_model.load_state_dict(torch.load(rec_model_dict, weights_only=True))
    rec_model.to(config["device"])

    atk_model = get_attacker(config, train_dataset).to(config["device"])

    config["trainer"].update({"optimizer": get_optimizer(config)})
    # Training
    trainer = get_atk_trainer(config)
    valid_metrics_values = trainer.fit(
        rec_model, atk_model, dataloaders, is_verbose=config.get("verbose", False)
    )
    # Testing
    best_atk_model_file = os.path.join(config["results_dir"], "best_model_utility.pt")
    atk_model.load_state_dict(torch.load(best_atk_model_file, weights_only=True))
    datasets, dataloaders = get_dataset_and_dataloader(config)
    test_metrics_values = trainer.test_epoch(rec_model, atk_model, dataloaders["test"])

    if config.get("verbose", False):
        print(valid_metrics_values)
        print(test_metrics_values)
    yaml_save(
        os.path.join(config["results_dir"], "test_metrics_results.yaml"),
        test_metrics_values,
    )
    return valid_metrics_values, test_metrics_values


def run_pairwise_atk_train_test(config: dict):
    reproducible(config["random_state"])
    splits = ["train", "val", "test"]
    config["dataset_config"]["splits"] = splits
    rec_datasets = get_rec_dataset(**config["dataset_config"])
    rec_train_dataset = rec_datasets[splits[0]]

    rec_model_conf = config.get("rec_model_config", None)
    rec_model_dict = config.get("rec_model_dict", None)
    rec_model = get_model(rec_model_conf, rec_train_dataset)
    rec_model.load_state_dict(torch.load(rec_model_dict, weights_only=True))
    rec_model.to(config["device"])
    datasets, dataloaders = get_atk_dataset_and_dataloader(config)
    train_dataset = datasets[splits[0]]
    atk_model = get_attacker(config, train_dataset).to(config["device"])

    config["trainer"].update({"optimizer": get_optimizer(config)})
    # Training
    trainer = get_atk_trainer(config)
    valid_metrics_values = trainer.fit(
        rec_model, atk_model, dataloaders, is_verbose=config.get("verbose", False)
    )
    # Testing
    best_atk_model_file = os.path.join(config["results_dir"], "best_model_utility.pt")
    atk_model.load_state_dict(torch.load(best_atk_model_file, weights_only=True))
    datasets, dataloaders = get_dataset_and_dataloader(config)
    test_metrics_values = trainer.test_epoch(rec_model, atk_model, dataloaders["test"])

    if config.get("verbose", False):
        print(valid_metrics_values)
        print(test_metrics_values)
    yaml_save(
        os.path.join(config["results_dir"], "test_metrics_results.yaml"),
        test_metrics_values,
    )
    return valid_metrics_values, test_metrics_values
