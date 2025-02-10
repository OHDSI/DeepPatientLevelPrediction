import time
import pathlib
import shutil
from urllib.parse import quote

import polars as pl
import duckdb
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data, labels=None, numerical_features=None):
        """
        data: path to the covariateData from Andromeda
        labels: a list of either 0 or 1, 1 if the patient got the outcome
        numerical_features: list of indices where the numerical features are
        """
        start = time.time()
        if pathlib.Path(data).suffix == ".sqlite":
            data = quote(data)
            self.data_ref = pl.read_database_uri(
                "SELECT * from covariateRef", uri=f"sqlite://{data}"
            ).lazy()
            data = pl.read_database_uri(
                "SELECT * from covariates", uri=f"sqlite://{data}"
            ).lazy()
        elif pathlib.Path(data).suffix == ".duckdb":
            data = quote(data)
            destination = pathlib.Path(data).parent.joinpath("python_copy.duckdb")
            path = shutil.copy(data, destination)
            conn = duckdb.connect(path)
            self.data_ref = conn.sql("SELECT * from covariateRef").pl().lazy()
            data = conn.sql("SELECT * from covariates").pl().lazy()
            # close connection
            conn.close()
            path.unlink()
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")
        observations = data.select(pl.col("rowId").max()).collect()[0, 0]
        # detect features are numeric
        if numerical_features is None:
            self.numerical_features = (
                data.group_by("columnId")
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
        cat_tensor = data_cat.to_torch()
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
        self.categorical_features = data_cat["columnId"].unique()
        
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
            indices = numerical_data.select(["rowId", "columnId"]).to_torch(dtype=pl.Int64)
            values = numerical_data.select("covariateValue").to_torch(dtype=pl.Float32)
            self.num = torch.sparse_coo_tensor(
                indices=indices.T,
                values=values.squeeze(),
                size=(observations, self.numerical_features.count()),
            ).to_dense()
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def get_feature_info(self):
        return {
            "numerical_features": len(self.numerical_features),
            "categorical_features": self.categorical_features.max(),
            "reference": self.data_ref
        }

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        if self.num is not None:
            batch = {"cat": self.cat[item, :], "num": self.num[item, :]}
        else:
            batch = {"cat": self.cat[item, :].squeeze(), "num": None}
        if batch["cat"].dim() == 1:
            batch["cat"] = batch["cat"].unsqueeze(0)
        if (batch["num"] is not None
                and batch["num"].dim() == 1
                and not isinstance(item, list)
        ):
            batch["num"] = batch["num"].unsqueeze(0)
        return [batch, self.target[item].squeeze()]





