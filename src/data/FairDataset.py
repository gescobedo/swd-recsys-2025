import os
import pandas as pd
from typing import List
import numpy as np
import torch
from src.data.BaseDataset import (
    UserRecDataset,
    BaseDataset,
    BaseAtkDataset,
    PairwiseRecDataset,
    EvalPairwiseDataset,
    PointwiseDataset,
    EvalPointwiseDataset,
)
from src.data.entity_feature import (
    InteractionFeature,
    FeatureDefinition,
)


class FairDataset(BaseAtkDataset):
    """
    Base fairness dataset class that all fair datasets should build upon
    """

    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition],
        split: str = "train",
        transform=None,
    ):
        super().__init__(data_dir, split, transform)
        # self.user_feature_names = [feat.name for feat in user_features]

        user_info = pd.read_csv(os.path.join(data_dir, split + "_user_features.csv"))
        # self.user_features = {
        #     feat.name: InteractionFeature(feat, user_info[feat.name])
        #     for feat in user_features
        # }
        self.has_user_features = user_features != None
        # self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, split + "_user_features.csv")
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }

    def __getitem__(self, idx):
        x_sample = super().__getitem__(idx)
        # feature_values = [
        #     self.user_features[feature.name].get_values()[idx]
        #     for feature in self.user_feature_list
        # ]
        if self.has_user_features:
            user_feature_values = [
                self.user_features[feature.name].get_values()[idx]
                for feature in self.user_features_list
            ]
        return idx, x_sample, user_feature_values  # TODO: generic dataloader?


class FeatureRecDataset(BaseDataset):
    """
    Base fairness dataset class that all fair datasets should build upon
    """

    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "train",
        dataset_type: str = "multi_hot",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample = super().__getitem__(idx)

        if self.has_user_features:
            user_feature_values = [
                self.user_features[feature.name].get_values()[
                    self.iterable_users.loc[idx]
                ]
                for feature in self.user_features_list
            ]
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[idx]
                for feature in self.item_features_list
            ]
        if self.has_user_features and self.has_item_features:
            return (idx, x_sample, y_sample, user_feature_values, item_feature_values)
        elif self.has_user_features:
            return (idx, x_sample, y_sample, user_feature_values)
        elif self.has_item_features:
            return (idx, x_sample, y_sample, item_feature_values)
        else:
            return idx, x_sample, y_sample


class FeatureUserRecDataset(UserRecDataset):

    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "train",
        dataset_type: str = "multi_hot",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            self.user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, self.user_info[feat.name])
                for feat in user_features
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample = super().__getitem__(idx)

        if self.has_user_features:
            user_feature_values = [
                self.user_features[feature.name].get_values()[
                    self.iterable_users.loc[idx]
                ]
                for feature in self.user_features_list
            ]
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[x_sample.astype(bool)]
                for feature in self.item_features_list
            ]

        if self.has_user_features and self.has_item_features:
            return (idx, x_sample, y_sample, user_feature_values, item_feature_values)
        elif self.has_user_features:
            return (idx, x_sample, y_sample, user_feature_values)
        elif self.has_item_features:
            return (idx, x_sample, y_sample, item_feature_values)
        else:
            return idx, x_sample, y_sample


class FeaturePairwiseRecDataset(PairwiseRecDataset):
    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "train",
        dataset_type: str = "pairwise",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }
            self.iterable_user_features = {
                name: feat.get_values()[self.iterable_users.index.values]
                for name, feat in self.user_features.items()
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample, neg_item = super().__getitem__(idx)
        user_feature_values = []
        if self.has_user_features:
            user_feature_values = [
                self.iterable_user_features[feature.name][x_sample]
                # self.user_features[feature.name].get_values()[
                #     self.iterable_users.loc[x_sample]
                # ]
                for feature in self.user_features_list
            ]
        item_feature_values = []
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[y_sample]
                for feature in self.item_features_list
            ]
        return (
            torch.tensor(x_sample),
            torch.tensor(y_sample),
            torch.tensor(neg_item),
            [],
            (user_feature_values, item_feature_values),
        )


class EvalFeaturePairwiseRecDataset(EvalPairwiseDataset):
    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "val",
        dataset_type: str = "pairwise",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample, _, exclude_items = super().__getitem__(idx)
        user_feature_values = []
        if self.has_user_features:
            user_feature_values = [
                self.user_features[feature.name].get_values()[
                    self.iterable_users.loc[x_sample]
                ]
                for feature in self.user_features_list
            ]
        item_feature_values = []
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[y_sample]
                for feature in self.item_features_list
            ]
        return (
            x_sample,
            y_sample,
            np.array([]).astype(int),
            exclude_items,
            (user_feature_values, item_feature_values),
        )


class FeaturePointwiseDataset(PointwiseDataset):
    def __init__(
        self,
        data_dir,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "train",
        dataset_type: str = "pointwise",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }
            self.iterable_user_features = {
                name: feat.get_values()[self.iterable_users.index.values]
                for name, feat in self.user_features.items()
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample = super().__getitem__(idx)
        user_feature_values = []
        if self.has_user_features:
            user_feature_values = [
                self.iterable_user_features[feature.name][x_sample]
                # self.user_features[feature.name].get_values()[
                #     self.iterable_users.loc[x_sample]
                # ]
                for feature in self.user_features_list
            ]
        item_feature_values = []
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[y_sample]
                for feature in self.item_features_list
            ]
        return (
            torch.tensor(x_sample),
            torch.tensor(y_sample),
            [],  # exclude-items for eval
            (user_feature_values, item_feature_values),
        )


class FeatureEvalPointwise(EvalPointwiseDataset):
    def __init__(
        self,
        data_dir: str,
        user_features: List[FeatureDefinition] = None,
        item_features: List[FeatureDefinition] = None,
        split: str = "val",
        dataset_type: str = "pointwise",
        transform=None,
    ):
        super().__init__(data_dir, split, dataset_type, transform)
        self.has_user_features = user_features != None
        self.has_item_features = item_features != None

        self.user_features_list = user_features
        if self.has_user_features:
            self.user_feature_names = [feat.name for feat in user_features]
            user_info = pd.read_csv(
                os.path.join(data_dir, "user_info.csv"), index_col="user_idx"
            )
            self.user_features = {
                feat.name: InteractionFeature(feat, user_info[feat.name])
                for feat in user_features
            }

        self.item_features_list = item_features
        if self.has_item_features:
            item_info = pd.read_csv(
                os.path.join(data_dir, "item_info.csv"), index_col="item_idx"
            )
            self.item_feature_names = [feat.name for feat in item_features]
            self.item_features = {
                feat.name: InteractionFeature(feat, item_info[feat.name])
                for feat in item_features
            }

    def __getitem__(self, idx):
        x_sample, y_sample, _, exclude_items = super().__getitem__(idx)
        user_feature_values = []
        if self.has_user_features:
            user_feature_values = [
                self.user_features[feature.name].get_values()[
                    self.iterable_users.loc[x_sample]
                ]
                for feature in self.user_features_list
            ]
        item_feature_values = []
        if self.has_item_features:
            item_feature_values = [
                self.item_features[feature.name].get_values()[y_sample]
                for feature in self.item_features_list
            ]
        return (
            x_sample,
            y_sample,
            np.array([]).astype(int),
            exclude_items,
            (user_feature_values, item_feature_values),
        )
