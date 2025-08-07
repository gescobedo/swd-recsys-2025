import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from src.recsys_models.BaseRecModel import BaseRecModel, PairWiseRecModel
from src.nn_modules.loss.losses import BPRLoss, EmbLoss
from src.utils.weight_init import general_weight_init
from src.data.BaseDataset import BaseDataset


class BPR(PairWiseRecModel):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, latent_size, n_users, n_items):
        super(BPR, self).__init__()
        self.embedding_size = latent_size
        self.n_users = n_users
        self.n_items = n_items
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calc_loss(self, user, pos_item, neg_item):
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        loss_dict = {"loss": loss}
        return loss, loss_dict

    def predict(self, user, item):
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_predict(self, user):
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    @staticmethod
    def build_from_config(model_config: dict, train_dataset: BaseDataset):
        model_config.update(
            {"n_users": train_dataset.n_users, "n_items": train_dataset.n_items}
        )
        return BPR(**model_config)

    def encode_user(self, x):
        return self.user_embedding(x.long())
