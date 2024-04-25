import time
import pathlib
import urllib

import polars as pl
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data, labels=None, numerical_features=None):
        """
        data: path to a covariates dataframe either arrow dataset or sqlite object
        labels: a list of either 0 or 1, 1 if the patient got the outcome
        numerical_features: list of indices where the numerical features are
        """
        start = time.time()
        if pathlib.Path(data).suffix == ".sqlite":
            data = urllib.parse.quote(data)
            data = pl.read_database(
                "SELECT * from covariates", connection=f"sqlite://{data}"
            ).lazy()
        else:
            data = pl.scan_ipc(pathlib.Path(data).joinpath("covariates/*.arrow"))
        observations = data.select(pl.col("rowId").max()).collect()[0, 0]
        # detect features are numeric
        if numerical_features is None:
            self.numerical_features = (
                data.groupby(by="columnId")
                .n_unique()
                .filter(pl.col("covariateValue") > 1)
                .select("columnId")
                .sort("columnId")
                .collect()["columnId"]
            )
        else:
            self.numerical_features = pl.Series("num", numerical_features)

        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        # filter by categorical columns,
        # select rowId and columnId
        data_cat = (
            data.filter(~pl.col("columnId").is_in(self.numerical_features))
            .select(pl.col("rowId"), pl.col("columnId"))
            .sort(["rowId", "columnId"])
            .with_columns(pl.col("rowId") - 1)
            .collect()
        )
        cat_tensor = torch.tensor(data_cat.to_numpy())
        tensor_list = torch.split(
            cat_tensor[:, 1],
            torch.unique_consecutive(cat_tensor[:, 0], return_counts=True)[1].tolist(),
        )

        # because of subjects without cat features, I need to create a list with all zeroes and then insert
        # my tensorList. That way I can still index the dataset correctly.
        total_list = [torch.as_tensor((0,))] * observations
        idx = data_cat["rowId"].unique().to_list()
        for i, i2 in enumerate(idx):
            total_list[i2] = tensor_list[i]
        self.cat = torch.nn.utils.rnn.pad_sequence(total_list, batch_first=True)
        self.cat_features = data_cat["columnId"].unique()

        # numerical data,
        # N x C, dense matrix with values for N patients/visits for C numerical features
        if self.numerical_features.count() == 0:
            self.num = None
        else:
            map_numerical = dict(
                zip(
                    self.numerical_features.sort().to_list(),
                    list(range(len(self.numerical_features))),
                )
            )

            numerical_data = (
                data.filter(pl.col("columnId").is_in(self.numerical_features))
                .sort("columnId")
                .with_columns(
                    pl.col("columnId").replace(map_numerical), pl.col("rowId") - 1
                )
                .select(
                    pl.col("rowId"),
                    pl.col("columnId"),
                    pl.col("covariateValue"),
                )
                .collect()
            )
            indices = torch.as_tensor(
                numerical_data.select(["rowId", "columnId"]).to_numpy(),
                dtype=torch.long,
            )
            values = torch.tensor(
                numerical_data.select("covariateValue").to_numpy(), dtype=torch.float
            )
            self.num = torch.sparse_coo_tensor(
                indices=indices.T,
                values=values.squeeze(),
                size=(observations, self.numerical_features.count()),
            ).to_dense()
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def get_numerical_features(self):
        return self.numerical_features

    def get_cat_features(self):
        return self.cat_features

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        if self.num is not None:
            batch = {"cat": self.cat[item, :], "num": self.num[item, :]}
        else:
            batch = {"cat": self.cat[item, :].squeeze(), "num": None}
        if batch["cat"].dim() == 1 and not isinstance(item, list):
            batch["cat"] = batch["cat"].unsqueeze(0)
        if (
            batch["num"] is not None
            and batch["num"].dim() == 1
            and not isinstance(item, list)
        ):
            batch["num"] = batch["num"].unsqueeze(0)
        return [batch, self.target[item].squeeze()]





