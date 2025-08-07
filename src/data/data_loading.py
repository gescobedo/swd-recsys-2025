import os
from typing import Union, List, Tuple, Dict
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader
from joblib.externals.loky.backend import get_context
from torch.utils.data import Dataset
from src.config.data_paths import get_data_path

from src.data.BaseDataset import BaseDataset, PairwiseRecDataset, EvalPairwiseDataset
from src.data.FairDataset import (
    FeatureRecDataset,
    FairDataset,
    FeatureUserRecDataset,
    FeaturePairwiseRecDataset,
    EvalFeaturePairwiseRecDataset,
)
from src.data.BaseDataloader import NegSamplingDataLoader

from src.data.entity_feature import FeatureDefinition
from scipy.sparse import csr_matrix, coo_matrix

# from src.data.BaseDataloader import TrainDataLoader
from src.config.config_enums import DatasetType


def sparse_scipy_to_tensor(matrix):
    return torch.sparse_coo_tensor(*sparse_scipy_to_tensor_params(matrix))


def sparse_scipy_to_tensor_params(matrix):
    # sparse tensor multiprocessing in dataloaders is not supported,
    # therefore we will create the sparse tensor only in training loop
    m = matrix.tocoo()
    indices = torch.stack([torch.tensor(m.row), torch.tensor(m.col)])
    return indices, m.data, m.shape


def sparse_tensor_to_sparse_scipy(tensor: torch.Tensor):
    return sp.coo_matrix((tensor._values(), tensor._indices()), shape=tensor.shape)


def train_collate_fn(data):
    # data must not be batched (not supported by PyTorch layers)
    indices, user_data, item_data, targets = data
    user_data = sparse_scipy_to_tensor_params(user_data)
    item_data = sparse_scipy_to_tensor_params(item_data)
    targets = torch.tensor(targets)
    return indices, user_data, item_data, targets


def train_collate_fn_fair(data):
    *data, traits = data
    return *train_collate_fn(data), torch.tensor(traits)


def get_atk_dataset(
    dataset: str,
    splits: List[str],
    dataset_type: str = DatasetType.multi_hot,
    user_features: List = None,
    item_features: List = None,
):
    datasets = {}
    user_features = [FeatureDefinition(**d) for d in user_features]
    data_path = get_data_path(dataset)
    for split in splits:
        dataset = FairDataset(
            data_dir=os.path.join(data_path, "0"),
            split=split,
            user_features=user_features,
            # item_features=item_features,
            transform=None,
        )
        datasets[split] = dataset
    return datasets


def get_rec_dataset(
    dataset: str,
    splits: List[str],
    dataset_type: str = DatasetType.multi_hot,
    user_features: List = None,
    item_features: List = None,
    transform=None,
):
    datasets = {}
    user_features = [FeatureDefinition(**d) for d in user_features]
    data_path = get_data_path(dataset)
    for split in splits:
        if dataset_type == DatasetType.pairwise:
            if split == "train":
                dataset = FeaturePairwiseRecDataset(
                    data_dir=data_path,
                    split=split,
                    dataset_type=dataset_type,
                    user_features=user_features,
                    item_features=item_features,
                    transform=transform,
                )
            else:
                dataset = EvalFeaturePairwiseRecDataset(
                    data_dir=data_path,
                    split=split,
                    dataset_type=dataset_type,
                    user_features=user_features,
                    item_features=item_features,
                    transform=transform,
                )
        else:
            dataset = FeatureUserRecDataset(
                data_dir=os.path.join(data_path),
                split=split,
                dataset_type=dataset_type,
                user_features=user_features,
                item_features=item_features,
                transform=transform,
            )
        datasets[split] = dataset

    return datasets


def get_dataloader(
    datasets: dict[str, BaseDataset],
    batch_size: Union[int, None] = 64,
    eval_batch_size: Union[int, None] = 64,
    n_workers: int = 0,
    shuffle_train=True,
    neg_sampling_strategy: str = "uniform",
    n_neg_samples: int = 1,
) -> dict[str, DataLoader]:

    dataloader_dict = {}
    for split, dataset in datasets.items():
        is_train_split = split == "train"
        if dataset.dataset_type == DatasetType.one_hot:
            raise NotImplementedError(
                f"Not supported dataset type {dataset.dataset_type}"
            )
        elif dataset.dataset_type == DatasetType.pairwise:
            if is_train_split:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size if split == "train" else eval_batch_size,
                    num_workers=n_workers,
                    shuffle=is_train_split and shuffle_train,
                    pin_memory=False,
                    # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
                )
                # loader = NegSamplingDataLoader(
                #     dataset=dataset,
                #     batch_size=batch_size if split == "train" else eval_batch_size,
                #     n_neg_samples=n_neg_samples,
                #     neg_sampling_strategy=neg_sampling_strategy,
                #     num_workers=n_workers,
                #     shuffle_train=is_train_split and shuffle_train,
                #     # pin_memory=True,
                #     # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
                # )
            else:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=eval_batch_size,
                    num_workers=n_workers,
                    shuffle=False,
                    # pin_memory=True,
                    # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
                )
        elif dataset.dataset_type == DatasetType.multi_one_hot:
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size if split == "train" else eval_batch_size,
                n_neg_samples=n_neg_samples,
                neg_sampling_strategy=neg_sampling_strategy,
                num_workers=n_workers,
                shuffle_train=is_train_split and shuffle_train,
                pin_memory=True,
                # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
            )
        else:
            # Default legacy mode multi_hot
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size if split == "train" else eval_batch_size,
                num_workers=n_workers,
                shuffle=is_train_split and shuffle_train,
                pin_memory=False,
                # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
            )
        dataloader_dict[split] = loader

    return dataloader_dict


def get_dataset_and_dataloader(
    config,
) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    datasets = get_atk_dataset(**config["dataset_config"])
    dataloader = get_dataloader(datasets, **config["trainer"]["dataloader"])
    return datasets, dataloader


def get_rec_dataset_and_dataloader(
    config,
) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    datasets = get_rec_dataset(**config["dataset_config"])
    dataloader = get_dataloader(datasets=datasets, **config["trainer"]["dataloader"])
    return datasets, dataloader


def get_atk_dataset_and_dataloader(
    config,
) -> Tuple[Dict[str, Dataset], Dict[str, DataLoader]]:
    datasets = get_atk_dataset(**config["dataset_config"])
    dataloader = get_dataloader(datasets=datasets, **config["trainer"]["dataloader"])
    return datasets, dataloader
