import wandb
import pandas as pd
from datetime import datetime


def wandb_write_to_run(run_name: str, key, data):
    api = wandb.Api()
    run = api.run(run_name)
    if isinstance(data, dict):
        for k, value in data.items():
            run.summary[f"{key}/{k}"] = value
    else:

        run.summary[f"{key}"] = data
    out = run.summary.update()


def export_sweep(entity, project, sweep_id, out_dir="./export"):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = api.runs(
        path=f"{entity}/{project}",
        order=sweep.order,
        filters={"$and": [{"sweep": f"{sweep_id}"}]},
    )
    rows = []
    for run in runs:
        row = {}
        row.update({"name": run.name})
        row.update(run.summary._json_dict)
        row.update(
            pd.json_normalize(
                {k: v for k, v in run.config.items() if not k.startswith("_")}, sep="."
            ).to_dict(orient="records")[0]
        )
        rows.append(row)
    runs_df = pd.DataFrame.from_records(rows)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_file = f"{now}--{entity}_{project}_{sweep_id}"
    runs_df.to_csv(f"{out_dir}/{run_name_file}.csv", index=False)
