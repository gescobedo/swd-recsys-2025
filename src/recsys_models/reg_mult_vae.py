from src.recsys_models.mult_vae import MultVAE
import torch
import torch.nn as nn
from src.nn_modules.loss.mmd import MMD
from src.nn_modules.loss.gsw import GSW
from typing import Dict, Tuple
import torch.nn.functional as F
from src.data.FairDataset import FairDataset
from src.utils.weight_init import general_weight_init


class RegMultVAE(
    MultVAE,
):
    def __init__(
        self,
        encoder_dims,
        decoder_dims=None,
        reg_params: str = None,
        reg_weight: float = 0.5,
        input_dropout: float = 0.5,
        activation_fn: str | nn.Module = nn.ReLU(),
        decoder_dropout=0,
        normalize_inputs=True,
        anneal_cap: float = 0.5,
        total_anneal_steps: int = 20000,
        l1_weight_decay=None,
    ):
        super().__init__(
            encoder_dims,
            decoder_dims,
            input_dropout,
            activation_fn,
            decoder_dropout,
            normalize_inputs,
            anneal_cap,
            total_anneal_steps,
            l1_weight_decay,
        )
        reg_type = reg_params["method"]
        self.reg = MMD() if reg_type == "mmd" else GSW(**reg_params)
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.apply(general_weight_init)

    def forward(self, x: torch.Tensor):
        z_sample, mu, logvar = self.encoder_forward(x)
        z = self.decoder_dropout(z_sample)
        z = self.decoder(z)
        return z, z_sample, mu, logvar

    def calc_loss(
        self, logits, targets, user_features
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        z, z_sample, mu, logvar = logits

        prob = F.log_softmax(z, dim=1)
        neg_ll = -torch.mean(torch.sum(prob * targets, dim=1))
        self.n_update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.n_update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_div = anneal * self._calc_KL_div(mu, logvar)
        # Dividing the batch acording to groups in this case binary
        # TODO : adaptation to >2 groups of users
        stat_loss = 0.0
        if self.reg_weight > 0:
            stat_loss = self.calc_stat_loss(z_sample, user_features)
        reg_loss = self.reg_weight * stat_loss
        loss = neg_ll + kl_div
        loss = loss + reg_loss

        loss_dict = {
            "nll": neg_ll,
            "KL": kl_div,
            f"{self.reg.__class__.__name__}_loss": reg_loss,
        }

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
            if (len(index_g0) > 0) and (len(index_g1) > 0):
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
    def build_from_config(model_config: dict, dataset: FairDataset):
        encoder_dims = model_config.get("encoder_dims", [])
        decoder_dims = model_config.get("decoder_dims", [])
        latent_size = model_config.get("latent_size")

        encoder_dims = [dataset.n_items] + encoder_dims
        encoder_dims = encoder_dims + [2 * latent_size]
        decoder_dims = [latent_size] + decoder_dims
        decoder_dims = decoder_dims + [dataset.n_items]
        model_config.pop("latent_size")
        model_config.pop("encoder_dims")
        model_config.pop("decoder_dims")
        return RegMultVAE(encoder_dims, decoder_dims, **model_config)
