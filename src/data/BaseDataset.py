import os
from scipy import sparse
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import scipy as sp
from scipy.sparse import csr_matrix, coo_matrix

# from src.config.config_enums import DatasetType


class BaseAtkDataset(Dataset):
    """
    Base dataset class that all datasets should build upon
    """

    def __init__(
        self,
        data_dir,
        split="train",
        dataset_type="multi_hot",
        transform=None,
    ):
        super().__init__()

        self.which = split
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.is_train_set = split == "train"

        # Determine input file and load data
        inputs_file_name = split + "_data.npz"
        self.data = sparse.load_npz(os.path.join(self.data_dir, inputs_file_name))

        # Determine target file and load data
        # targets_file_name = split + "_target.npz"

        # if self.is_train_set:
        #     # During training, we want to recreate the input
        #     self.targets = self.data
        # else:
        #     self.targets = sparse.load_npz(
        #         os.path.join(self.data_dir, targets_file_name)
        #     )

        self.n_users = self.data.shape[0]
        self.n_items = self.data.shape[1]
        self.n_interactions = self.data.sum()
        # if not self.is_train_set:
        #     self.n_interactions += self.targets.sum()

        self.__ensure_types()

    def __len__(self):
        return self.n_users

    def __ensure_types(self):
        self.data = self.data.astype("float32")
        # self.targets = self.targets.astype("float32")

    def __getitem__(self, idx):
        x_sample = self.data[idx, :].toarray().squeeze()
        if self.transform:
            x_sample = self.transform(x_sample)

        # y_sample = self.targets[idx, :].toarray().squeeze()

        return x_sample  # , y_sample


class RecDataset(Dataset):
    """
    Dataset to hold Recommender System data in the format of a pandas dataframe.
    """

    def __init__(self, data_path: str, split_set: str):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_idxs.csv, item_idxs.csv
        """
        assert split_set in [
            "train",
            "val",
            "test",
        ], f"<{split_set}> is not a valid value for split set!"
        self.data_path = data_path
        self.split_set = split_set

        self.n_users = None
        self.n_items = None

        self.user_to_user_group = None  # optional
        self.n_user_groups = None  # optional

        self.lhs = None

        self._load_data()

        self.name = "RecDataset"

    def _load_data(self):

        user_idxs = pd.read_csv(os.path.join(self.data_path, "user_idxs.csv"))
        item_idxs = pd.read_csv(os.path.join(self.data_path, "item_idxs.csv"))

        self.n_users = len(user_idxs)
        self.n_items = len(item_idxs)

        grouping_columns = [
            column for column in user_idxs.columns if column.endswith("group_idx")
        ]

        if len(grouping_columns) > 1:
            self.user_to_user_group = dict()
            self.n_user_groups = dict()
            for grouping_column in grouping_columns:
                grouping_column_name = grouping_column.split("_group_idx")[0]
                mapping = (
                    user_idxs[["user_idx", grouping_column]]
                    .set_index("user_idx")
                    .sort_index()[grouping_column]
                )
                mapping = torch.tensor(mapping)
                self.user_to_user_group[grouping_column_name] = mapping
                self.n_user_groups[grouping_column_name] = user_idxs[
                    grouping_column
                ].nunique()

        self.lhs = self._load_lhs(self.split_set)

    def _load_lhs(self, split_set: str):
        return pd.read_csv(os.path.join(self.data_path, f"{split_set}_data.csv"))

    def __len__(self):
        raise NotImplementedError(
            "RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
            "training or FullEvalDataset for evaluation."
        )

    def __getitem__(self, index):
        raise NotImplementedError(
            "RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
            "training or FullEvalDataset for evaluation."
        )


class TrainRecDataset(RecDataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interaction. It also stores the item popularity distribution over the training data.

    Additional notes:
    The data is loaded twice. Once the data is stored in a COO matrix to easily iterate over the dataset. Once in a CSR
    matrix to carry out fast negative sampling with the user-wise slicing functionalities (see also collate_fn in data/dataloader.py)
    """

    def __init__(self, data_path: str, delete_lhs: bool = True):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_idxs.csv, item_idxs.csv
        :param delete_lhs: Whether the pandas dataframe should be deleted after creating the iteration/sampling mtxs.
        """

        super().__init__(data_path, "train")

        self.delete_lhs = delete_lhs

        self.iteration_matrix = None
        self.sampling_matrix = None

        self.pop_distribution = None

        self._prepare_data()

    def _prepare_data(self):
        self.iteration_matrix = sparse.coo_matrix(
            (
                np.ones(len(self.lhs), dtype=np.int16),
                (self.lhs.user_idx, self.lhs.item_idx),
            ),
            shape=(self.n_users, self.n_items),
        )

        self.sampling_matrix = sparse.csr_matrix(self.iteration_matrix)

        item_popularity = np.array(self.iteration_matrix.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()

        if self.delete_lhs:
            del self.lhs

    def __len__(self):
        return self.iteration_matrix.nnz

    def __getitem__(self, index):
        user_idx = self.iteration_matrix.row[index].astype("int64")
        item_idx = self.iteration_matrix.col[index].astype("int64")

        return user_idx, item_idx, 1.0


class FullEvalDataset(RecDataset):
    """
    Dataset to hold Recommender System data and evaluate collaborative filtering algorithms. It allows iteration over
    all the users and compute the scores for all items (FullEvaluation). It also holds data from training and validation
    that needs to be excluded from the evaluation:
    During validation, items in the training data for a user are excluded as labels
    During test, items in the training data and validation for a user are excluded as labels
    """

    def __init__(self, data_path: str, split_set: str, delete_lhs: bool = True):
        """
        :param data_path: Path to the directory with listening_history_{val,test}.csv, user_idxs.csv, item_idxs.csv
        :param split_set: Either 'val' or 'test'
        :param delete_lhs: Whether the pandas dataframe should be deleted after creating the iteration/sampling mtxs.
        """

        super().__init__(data_path, split_set)

        self.delete_lhs = delete_lhs

        self.idx_to_user = None
        self.iteration_matrix = None
        self.exclude_data = None

        self._prepare_data()

        self.name = "FullEvalDataset"

    def _prepare_data(self):
        self.iteration_matrix = sparse.csr_matrix(
            (
                np.ones(len(self.lhs), dtype=np.int16),
                (self.lhs.user_idx, self.lhs.item_idx),
            ),
            shape=(self.n_users, self.n_items),
        )

        # Load Train data as well
        train_lhs = self._load_lhs("train")
        self.exclude_data = sparse.csr_matrix(
            (
                np.ones(len(train_lhs), dtype=bool),
                (train_lhs.user_idx, train_lhs.item_idx),
            ),
            shape=(self.n_users, self.n_items),
        )
        # If 'split_test' load also Valid data
        if self.split_set == "test":
            val_lhs = self._load_lhs("val")
            self.exclude_data += sparse.csr_matrix(
                (
                    np.ones(len(val_lhs), dtype=bool),
                    (val_lhs.user_idx, val_lhs.item_idx),
                ),
                shape=(self.n_users, self.n_items),
            )

        if self.delete_lhs:
            del self.lhs

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_index):
        return (
            user_index,
            self.iteration_matrix[user_index].toarray().squeeze().astype("float32"),
            self.exclude_data[user_index].toarray().squeeze().astype("bool"),
        )


class BaseDataset(Dataset):
    """
    Base dataset class that all datasets should build upon
    """

    def __init__(
        self,
        data_dir,
        split="train",
        dataset_type="multi_one_hot",
        transform=None,
    ):
        super().__init__()

        self.which = split
        self.data_dir = data_dir
        self.transform = transform
        self.dataset_type = dataset_type
        self.is_train_set = split == "train"
        self._load_data(split)
        self._construct_matrices()
        self.n_users = self.data.shape[0]
        self.n_items = self.data.shape[1]
        self.n_interactions = self.data.sum()

    def _load_data(self, split):
        self.user_idx = pd.read_csv(
            os.path.join(self.data_dir, f"user_idx.csv"), index_col="user_idx"
        )
        self.item_idx = pd.read_csv(
            os.path.join(self.data_dir, f"item_idx.csv"), index_col="item_idx"
        )
        if split == "train":
            data_src = pd.read_csv(os.path.join(self.data_dir, f"train_data.csv"))
            targets_src = data_src
        elif split == "val":
            data_src = pd.read_csv(os.path.join(self.data_dir, f"train_data.csv"))
            targets_src = pd.read_csv(os.path.join(self.data_dir, f"valid_data.csv"))
        elif split == "test":
            data_train = pd.read_csv(os.path.join(self.data_dir, f"train_data.csv"))
            data_valid = pd.read_csv(os.path.join(self.data_dir, f"valid_data.csv"))
            data_src = pd.concat([data_train, data_valid], axis=0)
            targets_src = pd.read_csv(os.path.join(self.data_dir, f"test_data.csv"))
        data_src.sort_values(["user_idx", "item_idx"], inplace=True)
        targets_src.sort_values(["user_idx", "item_idx"], inplace=True)
        data_src, targets_src = self._ensure_user_set(data_src, targets_src)
        self.data_src = data_src
        self.targets_src = targets_src

        self.iterable_users = (
            self.data_src["user_idx"].drop_duplicates().reset_index(drop=True)
        )

        self.iterable_user_index = self.iterable_users.copy()
        self.iterable_user_index.index.name = "new_user_idx"
        self.iterable_user_index = self.iterable_user_index.reset_index()
        self.iterable_user_index.set_index("user_idx", inplace=True)

        # Re indexing because of invalid users
        self.data_src["user_idx"] = self.iterable_user_index.loc[
            self.data_src["user_idx"].values, "new_user_idx"
        ].values
        self.targets_src["user_idx"] = self.iterable_user_index.loc[
            self.targets_src["user_idx"].values, "new_user_idx"
        ].values

    def _ensure_user_set(self, data: pd.DataFrame, target: pd.DataFrame):
        intersect_users = np.intersect1d(
            data["user_idx"].unique(), target["user_idx"].unique()
        )
        data = data[data["user_idx"].isin(intersect_users)]
        target = target[target["user_idx"].isin(intersect_users)]
        return data, target

    def _construct_matrices(self):

        uids_iids_array = self.data_src[["user_idx", "item_idx"]].values
        n_users, n_items = len(self.iterable_users), len(self.item_idx)
        data = np.ones(uids_iids_array.shape[0], dtype=np.int8)
        uids, iids = uids_iids_array[:, 0], uids_iids_array[:, 1]
        self.data = sparse.csr_matrix((data, (uids, iids)), (n_users, n_items))
        # reassigning invalid users

        uids_iids_array = self.targets_src[["user_idx", "item_idx"]].values
        # n_users, n_items = self.targets_src["user_idx"].nunique(), len(self.item_idx)
        data = np.ones(uids_iids_array.shape[0], dtype=np.int8)
        uids, iids = uids_iids_array[:, 0], uids_iids_array[:, 1]

        self.targets = sparse.csr_matrix((data, (uids, iids)), (n_users, n_items))
        # print(f"input:{str(self.data.shape)} target:{str(self.targets.shape)}")


class UserRecDataset(BaseDataset):

    def __init__(
        self, data_dir, split="train", dataset_type="multi_one_hot", transform=None
    ):
        super().__init__(data_dir, split, dataset_type, transform)

        if not self.is_train_set:
            self.n_interactions += self.targets.sum()

        self.__ensure_types()

    def __len__(self):
        return self.n_users

    def __ensure_types(self):
        self.data = self.data.astype("float32")
        self.targets = self.targets.astype("float32")

    @staticmethod
    def sparse_coo_to_tensor(coo: coo_matrix):
        """
        Transform scipy coo matrix to pytorch sparse tensor
        """
        values = coo.data
        indices = (coo.row, coo.col)  # np.vstack
        shape = coo.shape

        i = torch.LongTensor(indices)
        v = torch.DoubleTensor(values)
        s = torch.Size(shape)

        return torch.sparse.DoubleTensor(i, v, s)

    @staticmethod
    def sparse_batch_collate(batch):
        """
        Collate function which to transform scipy coo matrix to pytorch sparse tensor
        """
        # batch[0] since it is returned as a one element list
        data_batch, targets_batch = batch[0]

        if type(data_batch[0]) == csr_matrix:
            data_batch = data_batch.tocoo()  # removed vstack
            data_batch = UserRecDataset.sparse_coo_to_tensor(data_batch)
        else:
            data_batch = torch.DoubleTensor(data_batch)

        if type(targets_batch[0]) == csr_matrix:
            targets_batch = targets_batch.tocoo()  # removed vstack
            targets_batch = UserRecDataset.sparse_coo_to_tensor(targets_batch)
        else:
            targets_batch = torch.DoubleTensor(targets_batch)
        return data_batch, targets_batch

    def __getitem__(self, idx):

        x_sample = self.data[idx, :].toarray().squeeze()
        if self.transform:
            x_sample = self.transform(x_sample)

        y_sample = self.targets[idx, :].toarray().squeeze()

        return x_sample, y_sample


class PairwiseRecDataset(BaseDataset):
    def __init__(
        self, data_dir, split="train", dataset_type="pairwise", transform=None
    ):
        super().__init__(data_dir, split, dataset_type, transform)

        self._construct_sampling_matrix()
        self.sampling_row_indices = [
            self.sampling_matrix[i].indices for i in range(self.n_users)
        ]

        self.users_in_split = self.user_idx.index.values
        self.items_in_split = self.item_idx.index.values

    def _sample_negative(self, user):
        """Sample a negative item not interacted with by the user"""
        neg_item = np.random.randint(0, self.n_items)
        while (user, neg_item) in self.interaction_set:
            neg_item = np.random.randint(0, self.n_items)
        # self.interaction_set.add((user, neg_item))
        return neg_item

    def _construct_sampling_matrix(self):
        self.sampling_matrix = sparse.csr_matrix(self.data)
        item_popularity = np.array(self.data.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()
        self.data = sparse.coo_matrix(self.data)
        inter_rows, inter_cols = self.sampling_matrix.nonzero()
        self.interaction_set = set(zip(inter_rows, inter_cols))

    def __len__(self):
        return self.data.nnz

    def __getitem__(self, index):
        user_idx = self.data.row[index].astype("int64")
        item_idx = self.data.col[index].astype("int64")
        neg_item = self._sample_negative(user_idx)
        return user_idx, item_idx, neg_item


class EvalPairwiseDataset(PairwiseRecDataset):
    def __init__(self, data_dir, split="val", dataset_type="pairwise", transform=None):
        super().__init__(data_dir, split, dataset_type, transform)
        self.data = sparse.csr_matrix(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        user_index = self.iterable_users[index]
        pos_items = self.targets[user_index].toarray().squeeze().astype(int)
        return (
            user_index,
            pos_items,
            np.array([]),
            self.data[user_index].toarray().squeeze().astype("bool"),
        )


class PointwiseDataset(BaseDataset):
    def __init__(
        self, data_dir, split="train", dataset_type="pointwise", transform=None
    ):
        super().__init__(data_dir, split, dataset_type, transform)

    def __len__(self):

        return self.data.nnz

    def __getitem__(self, index):
        user_idx = self.data.row[index].astype("int64")
        item_idx = self.data.col[index].astype("int64")
        return user_idx, item_idx


class EvalPointwiseDataset(BaseDataset):
    def __init__(
        self, data_dir, split="train", dataset_type="pointwise", transform=None
    ):
        super().__init__(data_dir, split, dataset_type, transform)

    def __len__(self):

        return self.data.nnz

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        user_index = self.iterable_users[index]
        pos_items = self.targets[user_index].toarray().squeeze().astype(int)
        return (
            user_index,
            pos_items,
            np.array([]),
            self.data[user_index].toarray().squeeze().astype("bool"),
        )
