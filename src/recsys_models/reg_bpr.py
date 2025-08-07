import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from src.recsys_models.bpr import BPR
from src.nn_modules.loss.losses import BPRLoss, EmbLoss
from src.utils.weight_init import general_weight_init
from src.data.BaseDataset import BaseDataset
from src.nn_modules.loss.gsw import GSW
from src.nn_modules.loss.mmd import MMD


class RegBPR(BPR):

    def __init__(self, latent_size, n_users, n_items, reg_weight, reg_params):
        super(RegBPR, self).__init__(latent_size, n_users, n_items)
        self.embedding_size = latent_size
        self.n_users = n_users
        self.n_items = n_items
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.rec_weight = reg_weight
        reg_type = reg_params["method"]
        self.reg = MMD() if reg_type == "mmd" else GSW(**reg_params)
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

    def calc_loss(self, user, pos_item, neg_item, user_features):
        loss, loss_dict = super().calc_loss(user, pos_item, neg_item)
        user_embeddings = self.user_embedding(user)
        stat_loss = self.calc_stat_loss(user_embeddings, user_features)
        loss = loss + self.rec_weight * stat_loss
        return loss, loss_dict

    @staticmethod
    def build_from_config(model_config: dict, train_dataset: BaseDataset):
        model_config.update(
            {"n_users": train_dataset.n_users, "n_items": train_dataset.n_items}
        )
        return RegBPR(**model_config)
