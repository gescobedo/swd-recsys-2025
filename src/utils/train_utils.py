from typing import Union, Tuple, Dict
from numbers import Number
from src.utils.helper import save_model
import os
import re
import numpy as np
from collections import defaultdict


def postprocess_loss_results(result: Union[Number, Tuple[Number, Dict[str, Number]]]):
    """
    Utility function that ensures that the loss results have the format of
    (total_loss: float-like, sub_losses: dict[float-like])
    so that we can do easier processing (and logging) of results.
    """
    # we also want to allow regular loss functions that do not return additional partial losses
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    elif len(result) == 1:
        return result, {}
    else:
        raise SystemError("This type of loss result is not yet supported.")


def add_to_dict(original: dict, new: dict, multiplier: float = 1.0) -> dict:
    for k, l in new.items():
        original[k] += l * multiplier
    return original


def scale_dict_items(d: dict, scale: float) -> dict:
    return {k: v * scale for k, v in d.items()}


def dict_to_writer_format(d: dict):
    """
    Tensorboard does not allow lists for the params in its "add_hparams" function.
    Therefore, convert them to string instead.
    """
    writer_d = {}
    for k, v in d.items():
        writer_d[k] = (
            v if not isinstance(v, list) else "[{0:s}]".format(",".join(map(str, v)))
        )
    return writer_d


def adjust_results_dir(run_dir, change_fn):
    """
    Adjust the "train" (or other acceptable MODEL_DIRS) in a path like
    > results/ml-1m/<some_experiment>/0/train/<run_name>
    based on the change_fn function.
    """
    folders = ["train"]
    for d in folders:
        # glob pattern to regex pattern
        rd = d.replace("*", ".*?")
        regex_dir_sep = "\\" + os.path.sep
        rd = regex_dir_sep + "(" + rd + ")" + regex_dir_sep
        if res := re.search(rd, run_dir):
            dname = res[1]
            dname_replacement = change_fn(dname)

            # print(f"replacing '{dname}' with '{dname_replacement}'")

            # create own directory for results to keep folders nicely separated and clean
            return run_dir.replace(
                os.path.sep + dname + os.path.sep,
                os.path.sep + dname_replacement + os.path.sep,
            )
    return run_dir


def check_run_dict(run_dict, verbose=True):
    if len(run_dict) > 0:
        if verbose:
            print("Runs to attack are")
        for k in run_dict.keys():
            print(k)
        if verbose:
            print()
        return True

    if verbose:
        print("No (valid) runs found, canceling validation!\n")
    return False


def run_to_fold_dict(run_dict, verbose=True):
    fold_dict = defaultdict(lambda: dict())
    # split runs based on the fold they are in
    for run_dir, run_data in run_dict.items():
        fold_nr = extract_fold_nr(run_dir)
        if fold_nr is not None:
            fold_dict[fold_nr][run_dir] = run_data
        else:
            if verbose:
                print(
                    f"Could not determine on which fold to evaluate '{run_dir}' on. it is therefore ignored!"
                )
    return dict(fold_dict)


def extract_fold_nr(path):
    regex_dir_sep = "\\" + os.path.sep
    pattern = regex_dir_sep + r"(\d{1})" + regex_dir_sep
    if res := re.search(pattern, path):
        return int(res[1])
    return None
