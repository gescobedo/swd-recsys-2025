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


class MultVAE(BaseRecModel, VaeRecModel):
    def __init__(
        self,
        encoder_dims,
        decoder_dims=None,
        input_dropout: float = 0.5,
        activation_fn: Union[str, nn.Module] = nn.ReLU(),
        decoder_dropout=0.0,
        normalize_inputs=True,
        anneal_cap: float = 0.5,
        total_anneal_steps: int = 20000,
        l1_weight_decay=None,
    ):
        """
        Variational Autoencoders for Collaborative Filtering - Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara
        https://arxiv.org/abs/1802.05814
        Attributes
        ---------
        encoder_dims  : list
            list of values that defines the structure of the network on the decoder side
        decoder_dims : list
            list of values that defines the structure of the network on the encoder side (Optional)
        input_dropout: float
            dropout value
        """
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        assert (self.encoder_dims[-1] // 2) == self.decoder_dims[
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

        self.encoder = PolyLinear(
            self.encoder_dims, activation_fn, l1_weight_decay=l1_weight_decay
        )
        self.decoder = PolyLinear(
            self.decoder_dims, activation_fn, l1_weight_decay=l1_weight_decay
        )
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)
        # Annealing config
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.n_update = 0

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
        mu, logvar = x[:, : self.latent_size], x[:, self.latent_size :]
        z = self._sampling(mu, logvar)
        return z, mu, logvar

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encoder_forward(x)
        z = self.decoder_dropout(z)
        z = self.decoder(z)
        return z, mu, logvar

    def encode_user(self, x: torch.Tensor):
        z, _, _ = self.encoder_forward(x)
        return z

    def _sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std * self.training
        return z

    def _calc_KL_div(self, mu, logvar):
        """
        Calculates the KL divergence of a multinomial distribution with the generated
        mean and std parameters
        """
        # Calculation for multinomial distribution with multivariate normal and standard normal distribution based on
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # Mean is used as we may have different batch sizes, thus possibly have different losses throughout training
        return 0.5 * torch.mean(
            torch.sum(-logvar + torch.exp(logvar) + mu**2 - 1, dim=1)
        )

    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        z, mu, logvar = logits

        prob = F.log_softmax(z, dim=1)
        neg_ll = -torch.mean(torch.sum(prob * targets, dim=1))

        self.n_update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.n_update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_div = anneal * self._calc_KL_div(mu, logvar)

        loss = neg_ll + kl_div
        loss_dict = {"nll": neg_ll, "KL": kl_div}

        return loss, loss_dict

    @staticmethod
    def build_from_config(model_config: dict, dataset: BaseDataset):
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
        return MultVAE(encoder_dims, decoder_dims, **model_config)
