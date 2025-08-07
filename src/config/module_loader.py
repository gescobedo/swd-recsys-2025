from src.data.BaseDataset import BaseDataset
from src.trainer.BaseTrainer import BaseTrainer
from src.config.config_enums import (
    optim_class_choices,
    trainer_class_choices,
    atk_trainer_class_choices,
)
import torch
import importlib


def get_model(config: dict, dataset: BaseDataset) -> torch.nn.Module:
    """Retrieves the nn.Module child class according to config file model_class"""
    model_class = getattr(
        importlib.import_module("src.recsys_models"), config["model_class"]
    )
    model = model_class.build_from_config(config["model_config"], dataset)
    return model


def get_loss(config) -> torch.nn.Module:
    """Retrieves the nn.Module object child class according to config file model_class"""
    loss_class = getattr(
        importlib.import_module("src.modules.loss"), config["trainer"]["loss"]
    )
    return loss_class(config["trainer"]["loss_config"])(
        config["trainer"]["loss_config"]
    )


def get_optimizer(config: dict) -> torch.optim.Optimizer:
    """Retrieves the optimizer object according to config file"""
    return optim_class_choices[config["trainer"]["optimizer"]]


def get_trainer(config: dict) -> BaseTrainer:
    """Retrieves the trainer object according to config file"""
    trainer = trainer_class_choices[str.lower(config["model_class"])](config)
    # trainer.loss_fn = get_loss(config)
    return trainer


# Attacker Functions


def get_attacker(config: dict, dataset: BaseDataset) -> torch.nn.Module:
    """Retrieves the nn.Module child class according to config file model_class"""
    model_class = getattr(
        importlib.import_module("src.nn_modules"), config["model_class"]
    )
    model = model_class.build_from_config(config["model_config"], dataset)
    return model


def get_atk_trainer(config: dict) -> BaseTrainer:
    """Retrieves the trainer object according to config file"""
    trainer = atk_trainer_class_choices[str.lower(config["model_class"])](config)
    # trainer.loss_fn = get_loss(config)
    return trainer
