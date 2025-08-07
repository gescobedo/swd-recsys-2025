from typing import Optional
from enum import auto, Enum, StrEnum

import torch

from src.trainer import *


class DatasetSplitType(StrEnum):
    Random = auto()
    Temporal = auto()


class DatasetType(StrEnum):
    one_hot = auto()
    multi_hot = auto()
    multi_one_hot = auto()
    pairwise = auto()
    pointwise = auto()


class FeatureType(StrEnum):
    CATEGORICAL = auto()
    DISCRETE = auto()


class DatasetsEnum(StrEnum):

    ml1m = auto()
    lfmdemobias = auto()
    ambar = auto()


class OptimEnum(StrEnum):
    adam = auto()
    adamw = auto()
    sgd = auto()
    adadelta = auto()
    rmsprop = auto()


class AlgorithmsEnum(StrEnum):
    multvae = auto()
    advxmultvae = auto()
    vanillaae = auto()
    lightgcn = auto()
    bpr = auto()
    ncf = auto()

    regmultvae = auto()
    regvanillaae = auto()
    regae = auto()
    reglightgcn = auto()
    regbpr = auto()
    regncf = auto()


optim_class_choices = {
    OptimEnum.adam: torch.optim.Adam,
    OptimEnum.adamw: torch.optim.AdamW,
    OptimEnum.sgd: torch.optim.SGD,
    OptimEnum.adadelta: torch.optim.Adadelta,
    OptimEnum.rmsprop: torch.optim.RMSprop,
}
trainer_class_choices = {
    AlgorithmsEnum.multvae: AETrainer,
    AlgorithmsEnum.vanillaae: AETrainer,
    AlgorithmsEnum.regae: AETrainer,
    AlgorithmsEnum.regmultvae: UserFeatureAETrainer,
    AlgorithmsEnum.regvanillaae: UserFeatureAETrainer,
    AlgorithmsEnum.advxmultvae: AdvAETrainer,
    AlgorithmsEnum.lightgcn: PairwiseTrainer,
    AlgorithmsEnum.bpr: PairwiseTrainer,
    AlgorithmsEnum.reglightgcn: FeauturePairwiseTrainer,
    AlgorithmsEnum.regbpr: FeauturePairwiseTrainer,
    AlgorithmsEnum.ncf: FeauturePointwiseTrainer,
    AlgorithmsEnum.regncf: FeauturePointwiseTrainer,
}


class AttackerEnum(StrEnum):
    vaeembeddingattacker = auto()


atk_trainer_class_choices = {AttackerEnum.vaeembeddingattacker: AttackNetTrainer}
