from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Tuple, Dict, Union
from src.nn_modules.polylinear import PolyLinear, PolyLinearParallel
from src.data.FairDataset import FairDataset
from src.utils.weight_init import general_weight_init
from src.utils.helper import adjust_loss_params
from enum import StrEnum, auto


class LossesEnum(StrEnum):
    ce = auto()
    mse = auto()
    mmd = auto()
    mae = auto()


atk_loss_class_choices = {
    LossesEnum.ce: torch.nn.CrossEntropyLoss,
    LossesEnum.mse: torch.nn.MSELoss,
    LossesEnum.mae: torch.nn.L1Loss,
}


class MLPAttackerBaseModel(nn.Module, ABC):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @staticmethod
    @abstractmethod
    def build_from_config(self, config: Dict, train_dataset: FairDataset):
        pass

    @abstractmethod
    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class VaeEmbeddingAttacker(MLPAttackerBaseModel):
    def __init__(
        self,
        dims,
        n_parallel,
        input_dropout,
        loss_label,
        loss,
        feature,
        activation_fn: Union[str | nn.Module] = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.layers = PolyLinear(
            layer_config=dims,
            input_dropout=input_dropout,
            activation_fn=activation_fn,
        )
        self.loss = loss
        self.loss_label = loss_label
        self.feature = feature
        self.apply(general_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        loss = self.loss(logits, targets)
        loss_dict = {f"{self.feature}_{self.loss_label}_loss": loss}
        return loss, loss_dict

    def build_from_config(config: Dict, train_dataset: FairDataset):

        user_feature = train_dataset.user_features.get(config.get("feature"))
        layer_config = [config.get("embedding_size")] + config.get("dims")
        if user_feature.is_categorical_feature:
            layer_config = layer_config + [user_feature.n_unique_values]
            config.update(
                {
                    "loss": atk_loss_class_choices.get(config["loss"])(
                        weight=torch.tensor(
                            user_feature.class_weights, dtype=torch.float32
                        )
                    )
                }
            )
        elif user_feature.is_continuous_feature:
            layer_config = layer_config + [1]
            config.update({"loss": atk_loss_class_choices.get(config["loss"])})
        else:
            raise NotImplementedError("Not implemented type of feature")

        config.update({"loss_label": config.get("loss")})
        config.update({"dims": layer_config})
        config.pop("embedding_size")

        return VaeEmbeddingAttacker(**config)
