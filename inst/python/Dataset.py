import time
import pathlib
from urllib.parse import quote

import polars as pl
import torch
from torch.utils.data import Dataset

from pathlib import Path
import json
import os

# this one needs to have a parameter to not do the mapping again

class Data(Dataset):
    def __init__(self, data, labels=None, numerical_features=None,
                 in_cat_1_mapping=None, in_cat_2_mapping=None):
        desktop_path = Path.home() / "Desktop"

        desktop_path = Path.home() / "Desktop"
        with open(desktop_path / "data_path.txt", 'w') as f:
            f.write(data)
        with open(desktop_path / "labels.json", 'w') as f:
            json.dump(labels, f)

        # desktop_path = Path.home() / "Desktop"
        # with open(desktop_path / "data_path.txt", 'r') as f:
        #     data = f.read().strip()
        # with open(desktop_path / "labels.json", 'r') as f:
        #     labels = json.load(f)

        file_path = "/Users/henrikjohn/Desktop/poincare_model_dim_3.pt"
        state = torch.load(file_path)
        embed_names = state["names"]
        # print(f"Restored Names: {state['names']}")

        """
        data: path to a covariates dataframe either arrow dataset or sqlite object
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
        else:
            data = pl.scan_ipc(pathlib.Path(data).joinpath("covariates/*.arrow"))

        # # Fetch only the first few rows
        # data_head = data.limit(100).collect()
        # print("Head of the data:")
        # print(data_head)

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

        cat2_feature_names = []
        cat2_feature_names += embed_names

        # filter by categorical columns,
        # select rowId and columnId
        data_cat = (
            data.filter(~pl.col("columnId").is_in(self.numerical_features))
            .select(pl.col("rowId"), pl.col("covariateId"))
            .sort(["rowId", "covariateId"])
            .with_columns(pl.col("rowId") - 1)
            .collect()
        )

        # find concepts from the embedding that are available in the data
        cat2_ref = (
            self.data_ref
            .filter(pl.col("conceptId").is_in(cat2_feature_names))
            .select("covariateId")
            .collect()
        )

        # Now, use 'cat2_ref' as a normal DataFrame and access "columnId"
        data_cat_1 = data_cat.filter(
            ~pl.col("covariateId").is_in(cat2_ref["covariateId"]))
        
        self.cat_1_mapping = None
        if in_cat_1_mapping is None:
            self.cat_1_mapping = pl.DataFrame({
                "covariateId": data_cat_1["covariateId"].unique(),
                "index": pl.Series(range(1, len(data_cat_1["covariateId"].unique()) + 1))
            })
            # self.cat_1_mapping = pl.DataFrame(self.cat_1_mapping)
            self.cat_1_mapping.write_json(str(desktop_path / "cat1_mapping_train.json"))
        else:
            self.cat_1_mapping = pl.DataFrame(in_cat_1_mapping).with_columns(pl.col('index').cast(pl.Int64), pl.col('covariateId').cast(pl.Float64))
            self.cat_1_mapping.write_json(str(desktop_path / "cat1_mapping_test.json"))
        

        data_cat_1 = data_cat_1.join(self.cat_1_mapping, on="covariateId", how="left") \
            .select(pl.col("rowId"), pl.col("index").alias("covariateId"))

        cat_tensor = torch.tensor(data_cat_1.to_numpy())
        tensor_list = torch.split(
            cat_tensor[:, 1],
            torch.unique_consecutive(cat_tensor[:, 0], return_counts=True)[1].tolist(),
        )

        # because of subjects without cat features, I need to create a list with all zeroes and then insert
        # my tensorList. That way I can still index the dataset correctly.
        total_list = [torch.as_tensor((0,))] * observations
        idx = data_cat_1["rowId"].unique().to_list()
        for i, i2 in enumerate(idx):
            total_list[i2] = tensor_list[i]
        self.cat = torch.nn.utils.rnn.pad_sequence(total_list, batch_first=True)
        self.cat_features = data_cat_1["covariateId"].unique()

        # process cat_2 features
        data_cat_2 = data_cat.filter(
            pl.col("covariateId").is_in(cat2_ref))
        
        self.cat_2_mapping = None
        if in_cat_2_mapping is None:
            self.cat_2_mapping = pl.DataFrame({
                "covariateId": data_cat_2["covariateId"].unique(),
                "index": pl.Series(range(1, len(data_cat_2["covariateId"].unique()) + 1))
            })
            self.cat_2_mapping = self.cat_2_mapping.lazy()
            self.cat_2_mapping = (
                self.data_ref
                .filter(pl.col("covariateId").is_in(data_cat_2["covariateId"].unique()))
                .select(pl.col("conceptId"), pl.col("covariateId"))
                .join(self.cat_2_mapping, on="covariateId", how="left")
                .collect()
            )
            self.cat_2_mapping.write_json(str(desktop_path / "cat2_mapping_train.json"))
        else:
            self.cat_2_mapping = pl.DataFrame(in_cat_2_mapping).with_columns(pl.col('index').cast(pl.Int64), pl.col('covariateId').cast(pl.Float64))
            self.cat_2_mapping.write_json(str(desktop_path / "cat2_mapping_test.json"))
        
        # cat_2_mapping.write_json(str(desktop_path / "cat2_mapping.json"))

        data_cat_2 = data_cat_2.join(self.cat_2_mapping, on="covariateId", how="left") \
            .select(pl.col("rowId"), pl.col("index").alias("covariateId"))  # maybe rename this to something else

        cat_2_tensor = torch.tensor(data_cat_2.to_numpy())
        tensor_list_2 = torch.split(
            cat_2_tensor[:, 1],
            torch.unique_consecutive(cat_2_tensor[:, 0], return_counts=True)[
                1].tolist(),
        )

        total_list_2 = [torch.as_tensor((0,))] * observations
        idx_2 = data_cat_2["rowId"].unique().to_list()
        for i, i2 in enumerate(idx_2):
            total_list_2[i2] = tensor_list_2[i]
        self.cat_2 = torch.nn.utils.rnn.pad_sequence(total_list_2,
                                                     batch_first=True)
        self.cat_2_features = data_cat_2["covariateId"].unique()

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

    def get_cat_2_features(self):
        return self.cat_2_features
      
    def get_cat_2_mapping(self):
        return self.cat_2_mapping
    
    def get_cat_1_mapping(self):
        return self.cat_1_mapping

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        if self.num is not None:
            batch = {"cat": self.cat[item, :], "num": self.num[item, :], "cat_2": self.cat_2[item, :]}
        else:
            batch = {"cat": self.cat[item, :].squeeze(), "num": None, "cat_2": self.cat_2[item, :].squeeze(), }
        if batch["cat"].dim() == 1:
            batch["cat"] = batch["cat"].unsqueeze(0)
        if batch["cat_2"].dim() == 1:
            batch["cat_2"] = batch["cat_2"].unsqueeze(0)
        if (batch["num"] is not None
                and batch["num"].dim() == 1
                and not isinstance(item, list)
        ):
            batch["num"] = batch["num"].unsqueeze(0)
        return [batch, self.target[item].squeeze()]

