# %%
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import sparse as sp
from sklearn.model_selection import KFold, train_test_split
from typing import List, Tuple


def split_by_inter_ratio(
    data: pd.DataFrame,
    train_ratio=0.8,
    random_state=42,
    user_key: str = "userID",
):
    data = data.sort_values([user_key], ascending=True)
    cols = list(set(data.columns) - set(user_key))
    grouped = data.groupby(user_key)[cols].apply(
        (
            lambda x: x.sample(
                n=int(train_ratio * len(x)), random_state=random_state
            ).index.values
        ),
    )
    indexes_train = np.concatenate(grouped.values)
    data["tr"] = False
    data.loc[indexes_train, "tr"] = True

    train_data = data[data["tr"]]
    test_data = data[~data["tr"]]

    return train_data, test_data


def split_dataset(data, train_ratio=0.8, random_state=42, user_key="userID"):
    # train-test split
    train_data, test_data = split_by_inter_ratio(
        data=data, train_ratio=train_ratio, random_state=random_state, user_key=user_key
    )
    # train-valid split
    valid_data, test_data = split_by_inter_ratio(
        data=test_data,
        train_ratio=0.5,
        random_state=random_state,
        user_key=user_key,
    )
    return train_data, valid_data, test_data


def user_k_fold_split(
    data: pd.DataFrame,
    data_path,
    user_key="userID",
    item_key="itemID",
    item2token=None,
    user2token=None,
    n_folds=5,
    random_state=42,
):
    np.random.seed(random_state)
    # 'data' has to be already joined with all the wanted user features
    print(data["itemID"].nunique())
    interaction_matrix, user_info, item2token, user2token = (
        transform_dataframe_to_sparse(data, item2token=item2token)
    )

    # Saving origonal data
    sp.save_npz(os.path.join(data_path, "folds_interactions.npz"), interaction_matrix)
    user_info.to_csv(os.path.join(data_path, "folds_user_features.csv"), index=False)
    item2token.name = item_key
    item2token.to_csv(os.path.join(data_path, "folds_item2token.csv"), index=False)
    unique_users = data[user_key].unique()
    print(f"Split axis {user_key}, size : {len(unique_users)}")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for n_fold, (train_index, test_index) in enumerate(kf.split(unique_users)):
        print(
            f"Processing fold: {n_fold} \n(train, test):({len(train_index)}/{len(test_index)}) \ntotal: ({len(train_index)+len(test_index)})"
        )
        test_data = data[data[user_key].isin(unique_users[test_index])]
        remain_unique_users = unique_users[train_index]
        np.random.shuffle(remain_unique_users)
        samples_size = int(len(remain_unique_users) * 0.8)
        train_users, valid_users = (
            remain_unique_users[:samples_size],
            remain_unique_users[samples_size:],
        )
        valid_data = data[data[user_key].isin(valid_users)]
        train_data = data[data[user_key].isin(train_users)]
        fold_dir = os.path.join(data_path, str(n_fold))
        os.makedirs(fold_dir, exist_ok=True)
        #  selecting data
        train_data.to_csv(os.path.join(fold_dir, "user_train_data.csv"), index=False)
        valid_data.to_csv(os.path.join(fold_dir, "user_val_data.csv"), index=False)
        test_data.to_csv(os.path.join(fold_dir, "user_test_data.csv"), index=False)

        # creating sparse representations

        fold_train_im, fold_train_user_info, _, _ = transform_dataframe_to_sparse(
            interactions=train_data, item2token=item2token
        )
        fold_valid_im, fold_valid_user_info, _, _ = transform_dataframe_to_sparse(
            interactions=valid_data, item2token=item2token
        )
        fold_test_im, fold_test_user_info, _, _ = transform_dataframe_to_sparse(
            interactions=test_data, item2token=item2token
        )
        # print(fold_train_im.shape, fold_valid_im.shape, fold_test_im.shape)
        sp.save_npz(os.path.join(fold_dir, "train_data.npz"), fold_train_im)
        sp.save_npz(os.path.join(fold_dir, "val_data.npz"), fold_valid_im)
        sp.save_npz(os.path.join(fold_dir, "test_data.npz"), fold_test_im)

        # saving user features
        fold_train_user_info.to_csv(
            os.path.join(fold_dir, "train_user_features.csv"), index=False
        )
        fold_valid_user_info.to_csv(
            os.path.join(fold_dir, "val_user_features.csv"), index=False
        )
        fold_test_user_info.to_csv(
            os.path.join(fold_dir, "test_user_features.csv"), index=False
        )


def save_k_folds(
    data_folds: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], data_path: str
):
    for i, (train_data, valid_data, test_data) in enumerate(data_folds):
        fold_dir = os.path.join(data_path, str(i))
        os.makedirs(fold_dir, exist_ok=True)

        train_data.to_csv(os.path.join(fold_dir, "train_data.csv"), index=False)
        valid_data.to_csv(os.path.join(fold_dir, "val_data.csv"), index=False)
        test_data.to_csv(os.path.join(fold_dir, "test_data.csv"), index=False)


def read_dataset_to_obfuscate(data_dir: str):
    file_name = data_dir.split("/")[-1]
    train_data_url = f"{data_dir}/{file_name}.train.inter"
    valid_data_url = f"{data_dir}/{file_name}.valid.inter"
    test_data_url = f"{data_dir}/{file_name}.test.inter"
    dataset_name = file_name
    incl_data_url = f"{data_dir}/{file_name}_gender_incl.csv"

    train_data = transform_to_obf(pd.read_csv(train_data_url))
    valid_data = transform_to_obf(pd.read_csv(valid_data_url))
    test_data = transform_to_obf(pd.read_csv(test_data_url))
    user_data = train_data.drop_duplicates(["userID", "gender"]).reset_index()[
        ["userID", "gender"]
    ]
    inclination_data = pd.read_csv(incl_data_url, index_col="itemID")

    return train_data, valid_data, test_data, inclination_data, user_data, dataset_name


# %%
def transform_to_recbole(data):
    recbole_map = {
        "userID": "user_id:token",
        "itemID": "item_id:token",
        "gender": "gender:token",
        "timestamp": "timestamp:float",
        "rating": "rating:token",
        "freq": "freq:float",
        "tr": "tr:token",
    }
    data.rename(columns=recbole_map, inplace=True)
    return data


def transform_to_obf(data):
    recbole_map = {
        "user_id:token": "userID",
        "item_id:token": "itemID",
        "gender:token": "gender",
        "timestamp:float": "timestamp",
        "rating:token": "rating",
        "freq:float": "freq",
        "tr:token": "tr",
    }
    data.rename(columns=recbole_map, inplace=True)
    return data


# %%
def transform_dataframe_to_sparse(
    interactions: pd.DataFrame,
    item2token: pd.Series = pd.Series([]),
    user2token: pd.Series = pd.Series([]),
):
    """Generates a sparse matrix from a csv with user features

    Args:
        interactions (pd.DataFrame): _description_
        item2token (pd.Series, optional): _description_. Defaults to None.
        user2token (pd.Series, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if len(item2token) == 0:
        print(f"Creating item indexes")
        unique_items = interactions["itemID"].sort_values().unique()
        item2token = pd.Series(unique_items, name="itemID")
        item2token.index.name = "item_idx"
    token2item = pd.Series(data=item2token.index.values, index=item2token.values)
    if len(user2token) == 0:
        print(f"Creating user indexes")
        unique_users = interactions["userID"].unique()
        user2token = pd.Series(unique_users, name="userID")
        user2token.index.name = "user_idx"
    token2user = pd.Series(data=user2token.index.values, index=user2token.values)
    print(
        f"n_inter: {len(interactions)}: n_items: {len(item2token)} n_users: {len(user2token)}"
    )
    mapped_inter = interactions.copy()
    # assigning unique ids
    mapped_inter.loc[:, "userID"] = token2user.loc[mapped_inter["userID"]].values
    mapped_inter.loc[:, "itemID"] = token2item.loc[mapped_inter["itemID"]].values

    user_info = interactions.drop_duplicates(["userID"])

    uids_iids_array = mapped_inter[["userID", "itemID"]].values
    n_users, n_items = len(user2token), len(item2token)
    data = np.ones(uids_iids_array.shape[0], dtype=np.int8)

    uids, iids = uids_iids_array[:, 0], uids_iids_array[:, 1]
    interaction_matrix = sp.csr_matrix((data, (uids, iids)), (n_users, n_items))
    return interaction_matrix, user_info, item2token, user2token


### Preprocessing datasets
def read_ml1m(ROOT_DIR: str):
    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "ml-1m/ratings.dat"),
        sep="::",
        names=["userID", "itemID", "rating", "timestamp"],
        engine="python",
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "ml-1m/users.dat"),
        sep="::",
        names=["userID", "gender", "age", "occcupation", "zipcode"],
        engine="python",
    )
    return data_inter, data_user


def read_lfmdemobias(ROOT_DIR: str):
    # /lfm-demobias/sampled_100000_items_demo.txt
    # /lfm-demobias/sampled_100000_items_inter.txt
    # /lfm-demobias/sampled_100000_items_tracks.txt
    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias/sampled_100000_items_inter.txt"),
        sep="\t",
        header=None,
        names=["userID", "itemID", "pc"],
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias/sampled_100000_items_demo.txt"),
        delimiter="\t",
        names=["age", "gender"],
        usecols=[2, 3],
    )
    data_user["userID"] = data_user.index
    return data_inter, data_user


def read_ekstrabladet(ROOT_DIR: str):
    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "ekstrabladet/interactions.csv"),
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "ekstrabladet/user_features.csv"),
    )
    map_gender = {0.0: "M", 1.0: "F"}
    data_user["gender"] = data_user["gender"].apply(lambda x: map_gender.get(x))

    return data_inter, data_user


def read_lfmdb_small(ROOT_DIR: str):
    data_inter = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias/sampled_100000_items_inter.txt"),
        sep="\t",
        header=None,
        names=["userID", "itemID", "pc"],
    )
    data_user = pd.read_csv(
        os.path.join(ROOT_DIR, "lfm-demobias/sampled_100000_items_demo.txt"),
        delimiter="\t",
        names=["age", "gender"],
        usecols=[2, 3],
    )
    data_user["userID"] = data_user.index
    data_user = data_user.sample(1000, random_state=42)
    data_inter = data_inter[data_inter["userID"].isin(data_user.index.values)]
    return data_inter, data_user


def compute_item_features(inter_data: pd.DataFrame, item_features):
    item_features_list = []
    map_item_feat_function = {
        "item-stereo": lambda x: x,
        "popularity": lambda x: x,
    }
    for feat in item_features:
        item_features_list.append(map_item_feat_function[feat](inter_data))

    return pd.concat(item_features_list, axis=1)


def preprocess_dataset(
    data_inter: pd.DataFrame,
    data_user: pd.DataFrame,
    ROOT_DIR: str,
    dataset_name: str,
    K_CORE: int = 5,
):

    joined = data_inter.merge(data_user, on="userID").dropna()

    # Filtering dataset

    joined = core_filtering(joined, K_CORE)
    out_dir = os.path.join(ROOT_DIR, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Generate item features file
    # item_data = compute_item_features(joined, item_features)
    #
    print("Saving filtered dataset")

    interaction_matrix, user_info, item2token, user2token = (
        transform_dataframe_to_sparse(joined)
    )
    sp.save_npz(os.path.join(out_dir, "interactions.npz"), interaction_matrix)
    print(len(user_info), len(user2token))
    # mergin with ids
    item2token.to_csv(os.path.join(out_dir, "item_idx.csv"))
    user2token.to_csv(os.path.join(out_dir, "user_idx.csv"))
    user_idxs = user2token.reset_index()
    item_idxs = item2token.reset_index()
    user_info = user_info.merge(user_idxs)
    joined = joined.merge(user_idxs).merge(item_idxs)

    joined.to_csv(
        os.path.join(ROOT_DIR, dataset_name, f"{dataset_name}_filtered.csv"),
        index=False,
    )
    user_info.to_csv(os.path.join(out_dir, "user_info.csv"), index=False)
    # Spliting Dataset
    print("Generating recsys training data")
    rec_train_data, rec_valid_data, rec_test_data = split_dataset(joined)
    rec_train_data.to_csv(os.path.join(out_dir, "train_data.csv"), index=False)
    rec_valid_data.to_csv(os.path.join(out_dir, "valid_data.csv"), index=False)
    rec_test_data.to_csv(os.path.join(out_dir, "test_data.csv"), index=False)

    print("Generating k-folds for attackers")
    user_k_fold_split(rec_train_data, out_dir, item2token=item2token)


def get_stat_data(root, dataset):
    data_inter = pd.read_csv(os.path.join(root, dataset, f"{dataset}_filtered.csv"))
    user_info = pd.read_csv(os.path.join(root, dataset, f"user_info.csv"))
    print(
        f"n_inter: {len(data_inter)}: n_items: {data_inter['itemID'].nunique()} n_users: {data_inter['userID'].nunique()}"
    )
    print(user_info["gender"].value_counts().to_dict())


def core_filtering(data: pd.DataFrame, min_k: int = 5):
    while True:
        item_user_counts = data.groupby(["itemID"])["userID"].nunique().reset_index()
        user_item_counts = data.groupby(["userID"])["itemID"].nunique().reset_index()

        valid_items = item_user_counts[item_user_counts["userID"] >= min_k][
            "itemID"
        ].values
        valid_users = user_item_counts[user_item_counts["itemID"] >= min_k][
            "userID"
        ].values

        result = data[
            data["itemID"].isin(valid_items) & data["userID"].isin(valid_users)
        ]

        item_user_counts = (
            result.groupby(["itemID"])["userID"].nunique().reset_index()["userID"]
        )
        user_item_counts = (
            result.groupby(["userID"])["itemID"].nunique().reset_index()["itemID"]
        )

        if (item_user_counts >= min_k).any() and (user_item_counts >= min_k).any():
            break
        else:
            data = result
            print("Iterate filtering")
            if len(data) < 1000:
                print("invalid")
                break
    return result


ROOT_DIR = "/path/to/datasets/raw"
OUTPUT_DIR = "/path/to/datasets/processed"
lfm_data_small, lfm_user_data_small = read_lfmdb_small(ROOT_DIR)
print("*****************LFM-DEMOBIAS-1K*************************")
preprocess_dataset(
    lfm_data_small,
    lfm_user_data_small,
    ROOT_DIR=OUTPUT_DIR,
    dataset_name="lfm-demobias-1k",
)
lfm_data, lfm_user_data = read_lfmdemobias(ROOT_DIR)
print("*****************LFM-DEMOBIAS*************************")
preprocess_dataset(
    lfm_data, lfm_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="lfm-demobias"
)
print("*****************EKSTRABLADET*************************")
eb_data, eb_user_data = read_ekstrabladet(ROOT_DIR)
preprocess_dataset(
    eb_data, eb_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="ekstrabladet"
)
print("*****************ML 1M*************************")

ml1m_data, ml1m_user_data = read_ml1m(ROOT_DIR)
preprocess_dataset(ml1m_data, ml1m_user_data, ROOT_DIR=OUTPUT_DIR, dataset_name="ml-1m")

datasets = ["ml-1m", "lfm-demobias", "ekstrabladet"]
for d in datasets:
    print(d)
    get_stat_data(OUTPUT_DIR, d)

# %%
