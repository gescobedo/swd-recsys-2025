import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.config_classes.adv_config import AdvConfig
from src.config_classes.atk_config import AtkConfig


def accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    predictions = torch.argmax(inputs, dim=-1).detach().cpu().numpy()
    return torch.tensor(
        accuracy_score(targets.cpu().long().numpy(), predictions)
    ).item()


def balanced_accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    predictions = torch.argmax(inputs, dim=-1).detach().cpu().numpy()
    return torch.tensor(
        balanced_accuracy_score(targets.cpu().long().numpy(), predictions)
    ).item()


eval_fn_lookup = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    # need to flip order of inputs and targets for sklearn metrics
    "acc": accuracy,
    "bacc": balanced_accuracy,
}


def calculate_atk_metrics(eval_metrics: list, inputs, targets, return_individual=False):
    results_dict = {}
    for name in eval_metrics:
        results_dict.update({name: eval_fn_lookup[name](inputs, targets)})
    return results_dict
