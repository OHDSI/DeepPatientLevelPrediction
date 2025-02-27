import time
import pathlib
import shutil
from urllib.parse import quote

import polars as pl
import duckdb
import torch
from torch.utils.data import Dataset
from torch.nested import nested_tensor

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
            data_ref = pl.read_database_uri(
                "SELECT * from covariateRef", uri=f"sqlite://{data}"
            )
            analysis_ref = pl.read_database_uri(
                "SELECT * from analysisRef", uri=f"sqlite://{data}"
            )
            self.data_ref = data_ref.join(analysis_ref, on="analysisId").select(pl.all().exclude("collisions"))
            original_data = pl.read_database_uri(
                "SELECT * from covariates", uri=f"sqlite://{data}"
            ).lazy()
        elif pathlib.Path(data).suffix == ".duckdb":
            data = quote(data)
            destination = pathlib.Path(data).parent.joinpath("python_copy.duckdb")
            path = shutil.copy(data, destination)
            conn = duckdb.connect(path)
            data_ref = conn.sql("SELECT * from covariateRef").pl().lazy()
            analysis_ref = conn.sql("SELECT * from analysisRef").pl().lazy()
            self.data_ref = data_ref.join(analysis_ref, on="analysisId").select(pl.all().exclude("collisions"))
            original_data = conn.sql("SELECT * from covariates").pl().lazy()
            # close connection
            conn.close()
            path.unlink()
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")
        observations = original_data.select(pl.col("rowId").max()).collect()[0, 0]
        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        all_observations = (
            original_data.
            select("rowId").
            with_columns(pl.col('rowId') - 1).
            unique()
        )

        # filter by categorical columns,
        # select rowId and columnId
        # static data
        data = (
            original_data
            .select(pl.col("rowId"), pl.col("columnId"), pl.col("covariateValue"))
            .sort(["rowId", "columnId"])
            .with_columns(pl.col("rowId") - 1)
        )
        # the following gives me a dataframe with column rowId and then the sequence
        data = (
            data
            .group_by("rowId")
            .agg(
                pl.col("columnId").alias("feature_ids"),
                pl.col("covariateValue").alias("feature_values"),
            )
        )
        # left join with all_observations so all observations have an entry
        data = (
            all_observations
            .join(data, on="rowId", how="left")
            .sort("rowId")
            .with_columns(
                pl.col("feature_ids").fill_null(pl.lit([0], dtype=pl.List(pl.Int64))),
                pl.col("feature_values").fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
            )
            .collect()
        )

        self.data = {
            "row_ids": data["rowId"],
            "feature_ids": data["feature_ids"],
            "feature_values": data["feature_values"],
        }
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")


    def get_feature_info(self):
        return {
            "vocabulary_size": self.data_ref.select("columnId").max().collect(),
            "data_reference": self.data_ref.collect()
        }

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        batch = {
            "row_ids": self.data["row_ids"][item].to_torch(),
            "feature_ids": nested_tensor(self.data["feature_ids"][item].to_list(),
                                                  dtype=torch.long,
                                                  layout=torch.jagged).to_padded_tensor(padding=0),
            "feature_values": nested_tensor(self.data["feature_values"][item].to_list(),
                                                     dtype=torch.float32,
                                                     layout=torch.jagged).to_padded_tensor(padding=0),
        }
        return [batch, self.target[item].squeeze()]


class TemporalData(Dataset):
    def __init__(self, data, labels=None, numerical_features=None):
        """
        data: path to the covariateData from Andromeda
        labels: a list of either 0 or 1, 1 if the patient got the outcome
        numerical_features: list of indices where the numerical features are
        """
        start = time.time()
        if pathlib.Path(data).suffix == ".sqlite":
            data_path = quote(data)
            data_ref = pl.read_database_uri(
                "SELECT * from covariateRef", uri=f"sqlite://{data_path}"
            ).lazy()
            analysis_ref = pl.read_database_uri(
                "SELECT * from analysisRef", uri=f"sqlite://{data_path}"
            ).lazy()
            self.data_ref = data_ref.join(analysis_ref, on="analysisId").select(pl.all().exclude("collisions"))
            original_data = pl.read_database_uri(
                "SELECT * from covariates", uri=f"sqlite://{data_path}"
            ).lazy()
            self.time_ref = pl.read_database_uri(
                "SELECT * from timeRef", uri=f"sqlite://{data_path}"
            ).lazy()
        elif pathlib.Path(data).suffix == ".duckdb":
            data = quote(data)
            destination = pathlib.Path(data).parent.joinpath("python_copy.duckdb")
            path = shutil.copy(data, destination)
            conn = duckdb.connect(path)
            data_ref = conn.sql("SELECT * from covariateRef").pl().lazy()
            analysis_ref = conn.sql("SELECT * from analysisRef").pl().lazy()
            time_ref = conn.sql("SELECT * from timeRef").pl().lazy()
            self.data_ref = data_ref.join(analysis_ref, on="analysisId").select(pl.all().exclude("collisions"))
            self.time_ref = time_ref
            original_data = conn.sql("SELECT * from covariates").pl().lazy()
            # close connection
            conn.close()
            path.unlink()
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")
        observations = original_data.select(pl.col("rowId").max()).collect()[0, 0]

        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        all_observations = (
            original_data.
            select("rowId").
            with_columns(pl.col('rowId') - 1).
            unique()
        )

        # filter by categorical columns,
        # select rowId and columnId
        # static data
        data = (
            original_data
            .select(pl.col("rowId"), pl.col("columnId"), pl.col("timeId").cast(pl.Int64), pl.col("covariateValue"))
            .sort(["rowId", "columnId", "timeId"])
            .with_columns(pl.col("rowId") - 1,
                          pl.col("timeId").fill_null(-1))
        )
        # the following gives me a dataframe with column rowId and then the sequence
        data = (
            data
            .group_by("rowId")
            .agg(
                pl.col("columnId").sort_by("timeId").alias("feature_ids"),
                pl.col("covariateValue").sort_by("timeId").alias("feature_values"),
                pl.col("timeId").sort_by("timeId").alias("time_ids")
            )
        )
        # left join with all_observations so all observations have an entry
        data = (
            all_observations
            .join(data, on="rowId", how="left")
            .sort("rowId")
            .with_columns(
                pl.col("feature_ids").fill_null(pl.lit([0], dtype=pl.List(pl.Int64))),
                pl.col("feature_values").fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
                pl.col("time_ids").fill_null(pl.lit([-1], dtype=pl.List(pl.Int64)))
            )
            .collect()
        )

        self.data = {
            "row_ids": data["rowId"],
            "feature_ids": data["feature_ids"],
            "time_ids": data["time_ids"],
            "feature_values": data["feature_values"],
        }
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def get_feature_info(self):
        return {
            "vocabulary_size": self.data_ref.select("columnId").max().collect(),
            "data_reference": self.data_ref.collect(),
            "time_reference": self.time_ref.collect(),
        }

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        batch = {
            "row_ids": self.data["row_ids"][item].to_torch(),
            "feature_ids": nested_tensor(self.data["feature_ids"][item].to_list(),
                                                  dtype=torch.long,
                                                  layout=torch.jagged).to_padded_tensor(padding=0),
            "time_ids": nested_tensor(self.data["time_ids"][item].to_list(),
                                      dtype=torch.long,
                                      layout=torch.jagged).to_padded_tensor(padding=0),
            "feature_values": nested_tensor(self.data["feature_values"][item].to_list(),
                                                     dtype=torch.float32,
                                                     layout=torch.jagged).to_padded_tensor(padding=0),
        }
        return [batch, self.target[item].squeeze()]


