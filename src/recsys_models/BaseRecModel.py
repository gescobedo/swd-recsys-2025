from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Tuple, Dict, Union


class BaseRecModel(nn.Module, ABC):
    @abstractmethod
    def build_from_config(self):
        pass

    @abstractmethod
    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class AdvBaseRecModel(BaseRecModel):
    @abstractmethod
    def calc_adv_losses(self, adv_pred, adv_targets):
        pass

    @abstractmethod
    def build_adv_losses(self, adversary_config):
        pass

    @abstractmethod
    def build_adv_modules(self, adversary_config):
        pass


class VaeRecModel(ABC):
    @abstractmethod
    def encoder_forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PairWiseRecModel(BaseRecModel, ABC):

    @abstractmethod
    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def full_predict(self, x: torch.Tensor):
        pass


class StatRegModelMixin(nn.Module):
    def __init__(self, reg_type="gsw"):
        super().__init__()
        self.reg_type = reg_type

    def calc_stat_loss(self, mu: torch.Tensor, user_features: list[torch.Tensor]):
        group_labels = user_features[0]
        stat_loss = 0.0
        if self.reg_type == "mmd":
            stat_loss += self.reg(mu, group_labels)
        elif self.reg_type == "gswt":
            index_g0 = torch.argwhere((1 - group_labels)).flatten()
            index_g1 = torch.argwhere(group_labels).flatten()
            stat_loss += self.reg(mu[index_g0], mu[index_g1])
        elif self.reg_type == "gsw":
            index_g0 = torch.argwhere((1 - group_labels)).flatten()
            index_g1 = torch.argwhere(group_labels).flatten()
            g1 = mu[index_g1]
            g0 = mu[index_g0]
            if len(index_g0) > len(index_g1):
                n_rep = len(index_g0) // len(index_g1) + 1
                g1 = mu[index_g1].repeat(n_rep, 1)[: len(index_g0)]

            else:
                n_rep = len(index_g1) // len(index_g0) + 1
                g0 = mu[index_g0].repeat(n_rep, 1)[: len(index_g1)]

            stat_loss += self.reg(g0, g1)
        else:
            raise NotImplementedError(f"Stat-loss type {self.reg_type} not implemented")
        return stat_loss
