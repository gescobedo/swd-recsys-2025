from typing import Union, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from src.nn_modules.polylinear import PolyLinear
from src.data.BaseDataset import BaseDataset
from src.recsys_models.BaseRecModel import BaseRecModel, VaeRecModel
from src.nn_modules.loss.mmd import MMD
from src.nn_modules.loss.gsw import GSW
from src.utils.weight_init import general_weight_init


class VanillaAE(BaseRecModel):
    def __init__(
        self,
        encoder_dims,
        decoder_dims=None,
        input_dropout: float = 0.5,
        activation_fn: Union[str, nn.Module] = nn.ReLU(),
        decoder_dropout: float = 0.0,
        normalize_inputs=True,
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        assert (self.encoder_dims[-1]) == self.decoder_dims[
            0
        ], f"Latent dimensions of encoder and decoder networks do not match ({encoder_dims[-1]} vs {decoder_dims[0]})."
        assert self.encoder_dims[0] == self.decoder_dims[-1], (
            f"Input and output dimensions of encoder and decoder networks, respectively, "
            f"do not match ({encoder_dims[0]} vs {decoder_dims[-1]})."
        )

        self.n_items = self.encoder_dims[0]
        self.input_dropout = nn.Dropout(p=input_dropout)

        self.latent_size = self.decoder_dims[0]
        self.normalize_inputs = normalize_inputs

        self.encoder = PolyLinear(self.encoder_dims, activation_fn)
        self.decoder = PolyLinear(self.decoder_dims, activation_fn)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)

        self.apply(general_weight_init)

    def encoder_forward(self, x: torch.Tensor):
        """
        Performs the encoding step of the variational auto-encoder
        :param x: the unnormalized data to encode
        :return: the sampled encoding + the KL divergence of the generated mean and std params
        """
        x = self.input_dropout(x)
        if self.normalize_inputs:
            x = F.normalize(x, 2, 1)
        x = self.encoder(x)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encoder_forward(x)
        y = self.decoder_dropout(z)
        y = self.decoder(z)
        return y, z

    def encode_user(self, x: torch.Tensor):
        z = self.encoder_forward(x)
        return z

    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        y_pred, z = logits

        prob = F.log_softmax(y_pred, dim=1)
        neg_ll = -torch.mean(torch.sum(prob * targets, dim=1))

        loss = neg_ll
        loss_dict = {"nll": neg_ll}

        return loss, loss_dict

    @staticmethod
    def build_from_config(model_config: dict, dataset: BaseDataset):
        encoder_dims = model_config.get("encoder_dims", [])
        decoder_dims = model_config.get("decoder_dims", [])
        latent_size = model_config.get("latent_size")

        encoder_dims = [dataset.n_items] + encoder_dims
        encoder_dims = encoder_dims + [latent_size]
        decoder_dims = [latent_size] + decoder_dims
        decoder_dims = decoder_dims + [dataset.n_items]
        model_config.pop("latent_size")
        model_config.pop("encoder_dims")
        model_config.pop("decoder_dims")
        return VanillaAE(encoder_dims, decoder_dims, **model_config)


class RegVanillaAE(VanillaAE):
    def __init__(
        self,
        encoder_dims,
        reg_params: dict,
        reg_weight: float = 0.5,
        decoder_dims=None,
        input_dropout: float = 0.5,
        activation_fn: Union[str, nn.Module] = nn.ReLU(),
        decoder_dropout: float = 0.0,
        normalize_inputs=True,
    ):
        super().__init__(
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            input_dropout=input_dropout,
            activation_fn=activation_fn,
            decoder_dropout=decoder_dropout,
            normalize_inputs=normalize_inputs,
        )
        reg_type = reg_params["method"]
        self.reg = MMD() if reg_type == "mmd" else GSW(**reg_params)
        self.reg_type = reg_type
        self.reg_weight = reg_weight

        self.apply(general_weight_init)

    def encode_user(self, x: torch.Tensor):
        z = self.encoder_forward(x)
        return z

    def calc_loss(
        self, logits, targets, user_features
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        y_pred, z = logits

        prob = F.log_softmax(y_pred, dim=1)
        neg_ll = -torch.mean(torch.sum(prob * targets, dim=1))
        reg_loss = self.calc_stat_loss(z, user_features)
        loss = neg_ll + self.reg_weight * reg_loss
        loss_dict = {"nll": neg_ll, f"{self.reg.__class__.__name__}_loss": reg_loss}

        return loss, loss_dict

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

    @staticmethod
    def build_from_config(model_config: dict, dataset: BaseDataset):
        encoder_dims = model_config.get("encoder_dims", [])
        decoder_dims = model_config.get("decoder_dims", [])
        latent_size = model_config.get("latent_size")

        encoder_dims = [dataset.n_items] + encoder_dims
        encoder_dims = encoder_dims + [latent_size]
        decoder_dims = [latent_size] + decoder_dims
        decoder_dims = decoder_dims + [dataset.n_items]
        model_config.pop("latent_size")
        model_config.pop("encoder_dims")
        model_config.pop("decoder_dims")
        return RegVanillaAE(
            encoder_dims=encoder_dims, decoder_dims=decoder_dims, **model_config
        )
