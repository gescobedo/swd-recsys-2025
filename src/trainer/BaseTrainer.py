from typing import Tuple, Dict
from copy import deepcopy
from src.recsys_models.BaseRecModel import BaseRecModel
from torch.utils.data import DataLoader
from src.data.BaseDataset import BaseDataset
from src.logging.logger import Logger
import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.device = self.config["device"]
        self.dataset = self.config["dataset_config"]["dataset"]

        self.results_dir = self.config["results_dir"]

        self.trainer_config = self.config["trainer"]
        self.n_epochs = self.trainer_config["n_epochs"]
        self.eval_metrics = self.trainer_config.get("eval_metrics", None)
        self.metrics_top_k = self.trainer_config.get("metrics_top_k", None)
        self.early_stopping_criteria = self.trainer_config["early_stopping_criteria"]
        self.store_last_model = self.trainer_config["store_last_model"]

        self.store_model_every = self.trainer_config["store_model_every"]
        self.optimizer_class = self.trainer_config["optimizer"]

        self.logger = Logger(self.results_dir)
        self.use_wandb = self.config.get("use_wandb", False)
        self.log_every_n_batches = self.config["logging"]["log_every_n_batches"]

    @abstractmethod
    def fit(
        self,
        model: BaseRecModel,
        dataloaders: Dict[str, Tuple[BaseDataset, DataLoader]],
        include_test: bool = False,
        is_verbose: bool = False,
    ):
        pass

    @abstractmethod
    def run_epoch(self):
        pass

    @abstractmethod
    def validate_epoch(self):
        pass

    @abstractmethod
    def test_epoch(self):
        pass

    def _setup_optimizer(self, model: torch.nn.Module):
        self.optimizer = self.optimizer_class(
            model.parameters(), **self.trainer_config["optimizer_config"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.trainer_config["scheduler"]
        )

    def __str__(self):
        strconfig = []
        strconfig.append()
        strconfig.append(str(self.optimizer.param_groups))

        return "\n".join[strconfig]
