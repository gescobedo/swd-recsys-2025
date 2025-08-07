from src.utils.helper import (
    iter_configs,
    yaml_load,
    yaml_save,
    get_results_dir,
    save_experiment_config,
    reproducible,
    merge_dicts,
    get_devices,
    save_run_config,
)
from src.config.data_paths import get_storage_path
import json
import torch
from src.utils.input_validation import parse_input
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from typing import List, Dict
from src.config.module_loader import get_model, get_optimizer, get_trainer
from src.data.data_loading import (
    get_dataset_and_dataloader,
    get_rec_dataset_and_dataloader,
)


def get_wandb_conf(config: dict):

    base_config = yaml_load(config["base_config"])
    base_config["results_dir"] = os.path.join(
        config["base_dir"],
        config["dataset"],
        config["model"],
        f"{config['sweep']}/{config['run_id']}",
        "train",
    )
    base_config = merge_dicts(base_config, config)
    print("=" * 80)
    print("Final config is\n", json.dumps(base_config, indent=4))
    print("=" * 80)

    return base_config


def generate_job_configuration(input_config, results_dir):
    params = []

    devices = input_config["devices"]
    n_parallel = input_config["n_parallel"]
    devices = get_devices(devices, n_parallel)
    verbose = len(devices) == 1
    rnd = input_config.get("random_state", 42)
    seeds = input_config.get("random_state_seeds", [])
    if len(seeds) == 0:
        seeds = [rnd]

    for num_seed in range(len(seeds)):
        for conf, conf_name in iter_configs(input_config["config"]):
            conf.update(yaml_load(input_config["config"]))
            conf.update(
                {"results_dir": os.path.join(results_dir, str(num_seed), "train")}
            )
            # conf["dataset_config"].update(
            #     {
            #         "dataset": input_config["dataset"],
            #     }
            # )
            # conf["trainer"]["dataloader"].update(
            #     {"n_workers": input_config["n_workers"]}
            # )
            print(conf["trainer"]["dataloader"])
            conf["random_state"] = seeds[num_seed]
            conf["verbose"] = verbose
            params.append(conf)

    # Setting up devices
    current_device = 0
    for param in params:
        param.update({"device": devices[current_device]})
        current_device += 1

    return params


def setup_results_dir(input_config):
    # print(get_storage_path())
    results_dir_base = input_config.get("results_dir")
    if results_dir_base == None:
        results_dir_base = get_storage_path()
    results_dir = get_results_dir(
        input_config["dataset_config"]["dataset"],
        input_config["model_class"],
        results_dir_base,
    )
    os.makedirs(results_dir, exist_ok=False)
    return results_dir


def configure_experiment():
    input_config = parse_input(
        "experiments",
        options=[
            "gpus",
            "config",
            "n_parallel",
            "results_dir",
        ],
        access_as_properties=False,
    )
    exp_config = yaml_load(input_config["config"])
    exp_config.update(input_config)
    results_dir = setup_results_dir(exp_config)
    save_experiment_config(input_config["config"], results_dir)
    input_config.update({"results_dir": results_dir})
    params = generate_job_configuration(input_config, results_dir)

    return params


def run_experiments(params: List[Dict], n_devices: int, training_fn):
    print("Running", len(params), "training job(s)")
    results = Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
        delayed(training_fn)(p)
        for p in tqdm(params, desc="Running experiments", position=0, leave=True)
    )
    print("Done.")
    return results


def run_training(config):
    reproducible(config["random_state"])
    splits = ["train", "val"]
    config["dataset_config"]["splits"] = splits

    datasets, dataloaders = get_dataset_and_dataloader(config)
    train_dataset = datasets[splits[0]]

    model = get_model(config, train_dataset).to(config["device"])
    config["trainer"].update({"optimizer": get_optimizer(config)})
    trainer = get_trainer(config)
    metrics_values = trainer.fit(model, dataloaders)

    return metrics_values


def run_training_test(config):
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads // 2)
    reproducible(config["random_state"])
    save_run_config(config)

    splits = ["train", "val", "test"]
    config["dataset_config"]["splits"] = splits
    datasets, dataloaders = get_dataset_and_dataloader(config)
    train_dataset = datasets[splits[0]]
    model = get_model(config, train_dataset).to(config["device"])
    config["trainer"].update({"optimizer": get_optimizer(config)})
    # Training
    trainer = get_trainer(config)
    valid_metrics_values = trainer.fit(
        model, dataloaders, is_verbose=config.get("verbose", False)
    )
    # Testing
    datasets, dataloaders = get_dataset_and_dataloader(config)
    best_model_file = os.path.join(config["results_dir"], "best_model_utility.pt")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    test_metrics_values = trainer.test_epoch(model, dataloaders["test"])
    if config.get("verbose", False):
        print(valid_metrics_values)
        print(test_metrics_values)
    yaml_save(
        os.path.join(config["results_dir"], "test_metrics_results.yaml"),
        test_metrics_values,
    )
    return valid_metrics_values, test_metrics_values


def run_rec_training_test(config):
    num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads // 2)
    reproducible(config["random_state"])
    save_run_config(config)

    splits = ["train", "val", "test"]
    config["dataset_config"]["splits"] = splits
    datasets, dataloaders = get_rec_dataset_and_dataloader(config)
    train_dataset = datasets[splits[0]]

    model = get_model(config, train_dataset).to(config["device"])
    config["trainer"].update({"optimizer": get_optimizer(config)})
    # Training
    trainer = get_trainer(config)
    valid_metrics_values = trainer.fit(
        model, dataloaders, is_verbose=config.get("verbose", False)
    )
    # Testing
    datasets, dataloaders = get_rec_dataset_and_dataloader(config)
    best_model_file = os.path.join(config["results_dir"], "best_model_utility.pt")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    test_metrics_values = trainer.test_epoch(model, dataloaders["test"])
    if config.get("verbose", False):
        print(valid_metrics_values)
        print(test_metrics_values)
    yaml_save(
        os.path.join(config["results_dir"], "test_metrics_results.yaml"),
        test_metrics_values,
    )
    return valid_metrics_values, test_metrics_values


def run_test(config):
    splits = ["test"]
    config["dataset_config"]["splits"] = splits

    dataset_dataloader = get_dataset_and_dataloader(
        config["dataset_config"]["dataset"], config["trainer"]["dataloader"]
    )
    train_dataset = dataset_dataloader[splits[0]][0]
    model = get_model(config, train_dataset)
    trainer = get_trainer(config)
    metrics_values = trainer.test_epoch(model, dataset_dataloader)

    return metrics_values
