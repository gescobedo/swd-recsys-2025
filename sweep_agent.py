import os
import glob
import json
import wandb

from src.config.data_paths import get_storage_path
from src.config.config_exp import (
    get_wandb_conf,
    run_training_test,
    run_rec_training_test,
)


def train_val_agent():
    results_path = get_storage_path()

    # initialization and gathering hyperparameters 
    run = wandb.init(
        job_type="train/val/test/atk",
        allow_val_change=True,
        dir=results_path,
    )

    run_id = run.id
    project = run.project
    entity = run.entity
    sweep_id = run.sweep_id

    # retrieve config for run (this already contains the hyperparameter search modifications from W&B
    conf = {k: v for k, v in wandb.config.items() if k[0] != "_"}
    conf.update(
        {
            "base_dir": results_path,
            "entity": entity,
            "project": project,
            "sweep": sweep_id,
            "run_id": run_id,
        }
    )
    print(conf)

    print("=" * 80)
    print("W&B provided configuration is\n", json.dumps(conf, indent=4))
    print("=" * 80)

    # get full config
    conf = get_wandb_conf(conf)

    # updating wandb data
    run.tags += (conf["model_class"], conf["dataset_config"]["dataset"])

    # make wandb aware of the whole config we are using for the run
    # it is okay if this writes warnings about unsuccessful updates to stdout
    # (does so, even if actual values don't change)
    wandb.config.update(conf)
    run_rec_training_test(conf)

    print(f'W&B sweep ID is "{sweep_id}":offline={run.settings._offline}')

    keep_top_runs = 10
    # To reduce space consumption. Check if the run is in the top-10 best. If not, delete the model.
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    top_runs = api.runs(
        path=f"{entity}/{project}",
        per_page=keep_top_runs,
        order=sweep.order,
        filters={"$and": [{"sweep": f"{sweep_id}"}]},
    )[:keep_top_runs]
    top_runs_ids = {x.id for x in top_runs}

    if run_id not in top_runs_ids:
        print(
            f"Run {run_id} is not in the top-{keep_top_runs}." f"Model will be deleted"
        )

        # delete local run files
        alg_model_path = os.path.join(conf["results_dir"], "model.*")
        alg_model_files = glob.glob(alg_model_path)
        for alg_model_file in alg_model_files:
            os.remove(alg_model_file)

    wandb.finish()


if __name__ == "__main__":
    train_val_agent()
