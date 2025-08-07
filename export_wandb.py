# %
import pandas as pd
import wandb
import argparse


from src.utils.wandb_utils import export_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start an experiment")

    parser.add_argument(
        "--sweep_id",
        "-s",
        type=str,
        help="The W&B sweep id used to start the agents.",
    )

    parser.add_argument(
        "--out_dir",
        "-d",
        type=str,
        required=False,
        default="./export",
        help="destiny directory to save files",
    )

    args = parser.parse_args()
    fields = ["entity", "project", "sweep_id"]
    sweep_data = {k: v for k, v in zip(fields, args.sweep_id.split("/"))}
    sweep_data["out_dir"] = args.out_dir
    export_sweep(**sweep_data)
