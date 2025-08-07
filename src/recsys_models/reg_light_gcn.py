import numpy as np
import scipy.sparse as sp
import torch
from src.recsys_models.light_gcn import LightGCN
from src.nn_modules.loss.mmd import MMD
from src.nn_modules.loss.gsw import GSW
from src.data.BaseDataset import BaseDataset


class RegLightGCN(LightGCN):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(
        self,
        latent_size,
        interaction_matrix,
        n_users,
        n_items,
        n_layers,
        reg_weight=1e-05,
        require_pow=False,
        reg_params: dict = None,
        debias_reg_weight: float = 0.5,
    ):
        super(LightGCN, self).__init__(
            latent_size,
            interaction_matrix,
            n_users,
            n_items,
            n_layers,
            reg_weight,
            require_pow,
        )

        self.debias_rec_weight = debias_reg_weight
        reg_type = reg_params["method"]
        self.reg = MMD() if reg_type == "mmd" else GSW(**reg_params)
        self.reg_type = reg_type

    def calc_loss(self, user, pos_item, neg_item, user_features):
        loss, loss_dict = super().calc_loss(user, pos_item, neg_item)
        u_embeddings = self.user_embedding(user)
        debias_reg_loss = self.calc_stat_loss(u_embeddings, user_features)
        loss = loss + self.debias_rec_weight * debias_reg_loss

        new_loss_dict = {"loss": loss, "debias_reg_loss": debias_reg_loss}
        loss_dict.update(new_loss_dict)
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
    def build_from_config(model_config: dict, train_dataset: BaseDataset):
        model_config.update(
            {"n_users": train_dataset.n_users, "n_items": train_dataset.n_items}
        )
        return RegLightGCN(**model_config)
