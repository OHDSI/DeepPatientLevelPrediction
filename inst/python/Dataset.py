import time
from typing import Optional

import pathlib
import shutil
from urllib.parse import quote

import polars as pl
import pandas as pd # only for reticulate conversion of r dataframe
import duckdb
import torch
from torch.utils.data import Dataset
from torch.nested import nested_tensor

class Data(Dataset):
    def __init__(self, data, labels=None, data_reference=None):
        """
        data: path to the covariateData from Andromeda
        labels: a list of either 0 or 1, 1 if the patient got the outcome
        numerical_features: list of indices where the numerical features are
        """
        start = time.time()

        reader = DBReader(data)
        analysis_ref = reader.read_table("analysisRef", lazy = True)
        if data_reference is None:
            data_ref = reader.read_table("covariateRef", lazy = True)
        else:
            data_ref = pl.from_pandas(data_reference).lazy()
        self.data_ref = (data_ref
                         .join(analysis_ref, on="analysisId")
                         .select(pl.all().exclude("collisions")))
        original_data = reader.read_table("covariates", lazy=True)

        observations = original_data.select(pl.col("rowId").max()).collect()[0, 0]
        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        all_observations = (
            original_data
            .select("rowId")
            .with_columns(pl.col('rowId') - 1)
            .unique()
        )

        # select rowId and columnId
        # static data
        data = (
            original_data
            .select(pl.col("rowId"), pl.col("columnId"), pl.col("covariateValue"))
        )
        missing_means_zero = self.data_ref.filter(pl.col("missingMeansZero") == "Y").select("columnId")
        all_combinations = (
            all_observations
            .join(missing_means_zero, how="cross")
        )
        missing_means_zero = missing_means_zero.collect()["columnId"].to_list()
        data_mmz = (
            data
            .filter(pl.col("columnId").is_in(missing_means_zero))
            .join(all_combinations, on=["rowId", "columnId"], how="right")
            .with_columns(pl.col("covariateValue").fill_null(0.0))
            .select(["rowId", "columnId", "covariateValue"])
        )
        data_remaining = (
            data.filter(pl.col("columnId").is_in(missing_means_zero).not_())
        )
        data = (
            pl.concat([data_mmz, data_remaining])
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
        )


        self.data = {
            "row_ids": data.collect()["rowId"],
            "feature_ids": data.collect()["feature_ids"],
            "feature_values": data.collect()["feature_values"],
        }
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")


    def get_feature_info(self):
        return FeatureInfo(data_reference=self.data_ref.collect())

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
    def __init__(self, data: str, 
                 labels: Optional[list]=None, 
                 data_reference: Optional[pd.DataFrame]=None,
                 max_sequence_length: int=256, 
                 truncation: str="tail"):
        """
        Args:
            data: path to the covariateData from Andromeda
            labels: a list of either 0 or 1, 1 if the patient got the outcome
            data_reference: a DataFrame with the data reference
            max_sequence_length: the maximum sequence length
            truncation: the truncation strategy to use, only "tail" is implemented
        """
        start = time.time()
        reader = DBReader(data)

        analysis_ref = reader.read_table("analysisRef", lazy=True)
        if data_reference is None:
            data_ref = reader.read_table("covariateRef", lazy=True)
        else:
            data_ref = pl.from_pandas(data_reference).lazy()
        self.data_ref = data_ref.join(analysis_ref, on="analysisId").select(pl.all().exclude("collisions"))
        self.time_ref = reader.read_table("timeRef", lazy=True)
        original_data = reader.read_table("covariates", lazy=True)

        observations = original_data.select(pl.col("rowId").max()).collect()[0, 0]

        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        all_observations = (
            original_data.
            select("rowId").
            with_columns(pl.col('rowId') - 1). # R to Py
            unique()
        )

        missing_means_zero = self.data_ref.filter(pl.col("missingMeansZero") == "Y").select("columnId")
        all_combinations = (
            all_observations
            .join(missing_means_zero, how="cross")
        )
        missing_means_zero = missing_means_zero.collect()["columnId"].to_list()
        data = (
            original_data
            .select(pl.col("rowId"), pl.col("columnId"), pl.col("timeId").cast(pl.Int64), pl.col("covariateValue"))
            .with_columns(
                pl.col('rowId') - 1 # R to Py
                # pl.col("columnId") - 1 # R to Py
            )
        )

        data_mmz = (
            data
            .filter(pl.col("columnId").is_in(missing_means_zero))
            .join(all_combinations, on=["rowId", "columnId"], how="right")
            .with_columns(pl.col("covariateValue").fill_null(0.0))
            .select(["rowId", "columnId", "timeId", "covariateValue"])
        )
        data_remaining = (
            data.filter(pl.col("columnId")
                        .is_in(missing_means_zero)
                        .not_())
        )
        data = (
            pl.concat([data_mmz, data_remaining])
            .sort(["rowId", "columnId"])
            .with_columns(pl.col("rowId"))
        )
        # filter by categorical columns,
        # select rowId and columnId
        # static data
        data = (
            data
            .select(pl.col("rowId"), pl.col("columnId"), pl.col("timeId").cast(pl.Int64), pl.col("covariateValue"))
            .sort(["rowId", "columnId", "timeId"])
            .with_columns(pl.col("rowId"),
                          pl.col("timeId").fill_null(0))
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
                pl.col("time_ids").fill_null(pl.lit([0], dtype=pl.List(pl.Int64)))
            )
        )
        if max_sequence_length is not None:
            if truncation != "tail":
                raise NotImplementedError("Only tail truncation is implemented")
            data = data.with_columns([
                pl.col("feature_ids").list.eval(
                    pl.element().extend_constant(0, max_sequence_length).
                    slice(0, max_sequence_length)).alias(
                    "feature_ids"),
                pl.col("feature_values").list.eval(
                    pl.element().extend_constant(0.0, max_sequence_length).
                    slice(0, max_sequence_length)).alias(
                    "feature_values"),
                pl.concat_list(
                    pl.lit([0]),
         pl.col("time_ids").list.eval(
                    pl.element().extend_constant(0, max_sequence_length).
                    slice(0, max_sequence_length)))
                .alias("time_ids")
            ])

        data = data.collect(engine="streaming")


        self.data = {
            "row_ids": data["rowId"].to_torch(),
            "feature_ids": torch.as_tensor(data["feature_ids"].list.to_array(width=max_sequence_length).to_numpy(writable=True), dtype=torch.long),
            # +1 because time_ids don't have class token prepended later
            "time_ids": torch.as_tensor(data["time_ids"].list.to_array(width=max_sequence_length + 1).to_numpy(writable=True), dtype=torch.long),
            "feature_values": torch.as_tensor(data["feature_values"].list.to_array(width=max_sequence_length).to_numpy(writable=True), dtype=torch.float32),
        }
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def get_feature_info(self):
        return {
            "vocabulary_size": self.data_ref.select("columnId").max().collect().item(),
            "data_reference": self.data_ref.collect(),
            "time_reference": self.time_ref.collect(),
            "max_time_id": self.data["time_ids"].max().item() + 1
        }

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        batch = {
            "row_ids": self.data["row_ids"][item],
            "feature_ids": self.data["feature_ids"][item],
            "time_ids": self.data["time_ids"][item],
            "feature_values": self.data["feature_values"][item],
        }
        return [batch, self.target[item].squeeze()]


class DBReader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.suffix = pathlib.Path(db_path).suffix
        self.data_quoted = quote(db_path)

    def read_table(self, table: str, *, lazy: bool=False) -> pl.DataFrame | pl.LazyFrame:
        query = f"SELECT * FROM {table}"

        if self.suffix == ".sqlite":
            df = pl.read_database_uri(query, uri=f"sqlite://{self.data_quoted}")
        elif self.suffix == ".duckdb":
            dest_path = pathlib.Path(self.data_quoted).parent.joinpath("python_copy.duckdb")
            temp_path = pathlib.Path(shutil.copy(self.data_quoted, dest_path))
            conn = duckdb.connect(str(temp_path))
            df = conn.sql(query).pl()
            conn.close()
            temp_path.unlink()
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")
        return df.lazy() if lazy else df


class FeatureInfo(object):
    def __init__(self, data_reference: pl.DataFrame, time_reference: Optional[pl.DataFrame] = None):
        self.data_reference = data_reference
        self.time_reference = time_reference

    def get_vocabulary_size(self) -> int:
        return int(
            self.data_reference
            .filter(pl.col("isBinary") == "Y")
            .select("columnId")
            .max()
            .item())

    def get_numerical_feature_ids(self) -> torch.Tensor:
        return (
            self.data_reference
            .filter(pl.col("isBinary") == "N")
            .select("columnId")
            .sort("columnId")
            .to_torch()
            .squeeze(1)
        )

    def get_max_time_id(self):
        if self.time_reference is None:
            return None
        return int(
            self.time_reference
            .select("timeId")
            .max()
            .item() + 1
        )

