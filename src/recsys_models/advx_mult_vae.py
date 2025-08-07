from typing import Union, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from src.nn_modules.polylinear import PolyLinear
from src.nn_modules.adversary import Adversary
from src.nn_modules.parallel import Parallel, ParallelMode
from src.data.FairDataset import FairDataset
from src.recsys_models.BaseRecModel import (
    BaseRecModel,
    AdvBaseRecModel,
    VaeRecModel,
)
from enum import StrEnum, auto
from src.utils.weight_init import general_weight_init

from src.config_classes.adv_config import AdvConfig


class AdvLossEnum(StrEnum):
    mse = auto()
    mae = auto()
    ce = auto()


loss_type_mapping = {
    AdvLossEnum.mse: torch.nn.MSELoss,
    AdvLossEnum.mae: torch.nn.L1Loss,
    AdvLossEnum.ce: torch.nn.CrossEntropyLoss,
}


class AdvXMultVAE(AdvBaseRecModel, VaeRecModel):
    def __init__(
        self,
        encoder_dims,
        decoder_dims=None,
        adversary_config=None,
        input_dropout: float = 0.5,
        activation_fn: Union[str, nn.Module] = nn.ReLU(),
        decoder_dropout=0.0,
        normalize_inputs=True,
        anneal_cap: float = 0.5,
        total_anneal_steps: int = 20000,
        l1_weight_decay=None,
    ):
        """ """
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
        # Advesarial  config
        self.build_adv_modules(adversary_config)
        self.build_adv_losses(adversary_config)
        self.adversaries_enabled = (
            adversary_config is not None and len(adversary_config) > 0
        )
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
        adv_results = []
        if self.adversaries_enabled:
            adv_results = self.adversaries(z)
        z = self.decoder_dropout(z)
        z = self.decoder(z)
        return z, mu, logvar, adv_results

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
        self, logits, targets, user_features
    ) -> Tuple[torch.Tensor | Dict[str, torch.Tensor]]:
        # MultVAE Loss
        z, mu, logvar, adv_pred = logits
        rec_targets = targets
        prob = F.log_softmax(z, dim=1)
        neg_ll = -torch.mean(torch.sum(prob * rec_targets, dim=1))

        self.n_update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.n_update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_div = anneal * self._calc_KL_div(mu, logvar)

        loss = neg_ll + kl_div
        loss_dict = {"nll": neg_ll, "KL": kl_div}

        # Adversarial Loss
        adv_loss, adv_losses_dict = self.calc_adv_losses(adv_pred, user_features)
        loss += adv_loss
        loss_dict.update(adv_losses_dict)

        return loss, adv_loss, loss_dict

    def calc_adv_losses(self, adv_pred, adv_targets):
        adv_loss = 0.0
        adv_losses_dict = {}
        for logits, targets, (feature, adv_module_loss) in zip(
            adv_pred, adv_targets, self.adv_losses.items()
        ):
            module_adv_loss = 0.0
            # Calculates mean of the calculated loss of the parallel modules per features
            # Look into the configuration of the module this loop should be as equal as the n_paralallel parameter
            for single_logits in logits:
                module_adv_loss += adv_module_loss(single_logits, targets)
            mean_module_adv_loss = module_adv_loss / len(logits)
            module_key = f"adv_loss/{feature}_{adv_module_loss.__class__.__name__}"
            adv_losses_dict[module_key] = mean_module_adv_loss
            adv_loss += mean_module_adv_loss
        return adv_loss, adv_losses_dict

    def build_adv_losses(self, adversary_config):
        self.adv_losses = OrderedDict()
        for config in adversary_config:
            self.adv_losses[config["feature"]] = config["loss"]

    def build_adv_modules(self, adversary_config):
        modules = []
        for config in adversary_config:
            modules.append(Adversary(AdvConfig(**config)))

        # pack into module list to register values as parameters
        self.adversaries = Parallel(
            modules=modules, parallel_mode=ParallelMode.SingleInMultiOut
        )

    @staticmethod
    def build_from_config(model_config: dict, dataset: FairDataset):
        encoder_dims = model_config.get("encoder_dims", [])
        decoder_dims = model_config.get("decoder_dims", [])
        latent_size = model_config.get("latent_size")

        encoder_dims = [dataset.n_items] + encoder_dims
        encoder_dims = encoder_dims + [2 * latent_size]
        decoder_dims = [latent_size] + decoder_dims
        decoder_dims = decoder_dims + [dataset.n_items]

        adv_config = model_config["adversary_config"]
        adv_modules_config = []
        for feature_config in adv_config:
            feaure_name = feature_config["feature"]
            user_feature = dataset.user_features.get(feature_config.get("feature"))
            module_dims = [latent_size] + feature_config.get("dims")
            if user_feature.is_categorical_feature:
                module_dims = module_dims + [user_feature.n_unique_values]
                loss_module = loss_type_mapping.get(feature_config["loss"])(
                    weight=torch.tensor(
                        user_feature.class_weights, dtype=torch.float32
                    ).cuda()
                )
            elif user_feature.is_continuous_feature:
                layer_config = layer_config + [1]
                loss_module = loss_type_mapping.get(feature_config["loss"])
            else:
                raise NotImplementedError("Not implemented type of feature")
            feature_config.update({"loss": loss_module})
            feature_config.update({"dims": module_dims})
            feature_config.update(
                {"type": dataset.user_features.get(feaure_name).feature.type}
            )
            adv_modules_config.append(feature_config)

        model_config.update({"adversary_config": adv_modules_config})
        model_config.pop("latent_size")
        model_config.pop("encoder_dims")
        model_config.pop("decoder_dims")
        return AdvXMultVAE(encoder_dims, decoder_dims, **model_config)
