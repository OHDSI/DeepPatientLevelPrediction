import shutil
import time
from typing import Optional, Union, Sequence

import pathlib
from urllib.parse import quote
import uuid

import polars as pl
import duckdb
import torch
from polars import LazyFrame
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(
        self,
        data_path: str,
        labels: Optional[list] = None,
        data_reference: Optional[pl.DataFrame | pl.LazyFrame] = None,
        temporal_settings: Optional[dict] = None,
    ):
        """
        Args:
            data_path: path to the covariateData Andromeda object
            labels: a list of either 0 or 1, 1 if the patient got the outcome
            data_reference: a DataFrame with the data reference
            temporal_settings: a dictionary with the following keys:
                time_tokens: whether to insert time tokens or not
                truncation: the truncation strategy to use, only "tail" is implemented
        """
        start = time.time()

        data_dict = read_data(data_path, lazy=True, data_reference=data_reference)
        self.use_time = "timeId" in data_dict["data"].collect_schema().names()

        data = fill_missing_means_zero(
            data_dict["data"], data_dict["data_ref"], with_time=self.use_time
        )

        if (
                self.use_time
                and temporal_settings is not None
                and temporal_settings["time_tokens"]
        ):
            data, data_ref = insert_time_tokens(
                data, data_ref=data_dict["data_ref"], strategy="coarse"
            )
        else:
            data_ref = data_dict["data_ref"]

        data = aggregate_sequences(data, with_time=self.use_time)


        if self.use_time and temporal_settings is not None:
            max_sequence_length = temporal_settings["max_sequence_length"]
            if max_sequence_length == "max":
                max_sequence_length = (
                    data
                    .select(pl.col("feature_ids").list.len().max())
                    .collect()
                    .item()
                )
                print(f"Using max sequence length: {max_sequence_length}")
            truncation = temporal_settings["truncation"]
        else:
            max_sequence_length = (
                data
                .select(pl.col("feature_ids").list.len().max())
                .collect()
                .item()
            )
            truncation = "tail"

        data = pad_prefix_all(
            data,
            max_sequence_length=max_sequence_length,
            truncation=truncation,
            use_time=self.use_time,
        )

        data = data.sort(by="rowId").collect(engine="streaming")

        self.data = {
            "row_ids": data["rowId"].to_torch(),
            "feature_ids": torch.as_tensor(
                data["feature_ids"]
                .list.to_array(width=max_sequence_length)
                .to_numpy(writable=True),
                dtype=torch.long,
            ),
            "feature_values": torch.as_tensor(
                data["feature_values"]
                .list.to_array(width=max_sequence_length)
                .to_numpy(writable=True),
                dtype=torch.float32,
            ),
        }
        if self.use_time:
            # +1 because time_ids don't have class token prepended later
            self.data["time_ids"] = torch.as_tensor(
                data["time_ids"]
                .list.to_array(width=max_sequence_length + 1)
                .to_numpy(writable=True),
                dtype=torch.long,
            )
        self.data_ref = data_ref
        self.time_ref = data_dict["time_ref"] if self.use_time else None

        self.target = torch.tensor(
            labels if labels is not None else [0] * len(data), dtype=torch.float32
        )
        delta = time.time() - start
        print(f"Processed data in {delta:.2f} seconds")

    def get_feature_info(self):
        feature_info = FeatureInfo(
            data_reference=self.data_ref.collect(),
            time_reference=self.time_ref.collect()
            if self.time_ref is not None
            else None,
        )
        if self.use_time:
            # +1 because timeId starts from 0
            feature_info.set_max_time_id(self.data["time_ids"].max().item() + 1)
        return feature_info

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        batch = {
            "feature_ids": self.data["feature_ids"][[item]],
            "time_ids": self.data["time_ids"][[item]] if self.use_time else None,
            "feature_values": self.data["feature_values"][[item]],
        }
        return [batch, self.target[item].squeeze()]


class DBReader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.suffix = pathlib.Path(db_path).suffix
        if self.suffix == "":
            # unzip to a new location
            new_db_path = (
                pathlib.Path(db_path).parent.joinpath(f"db_{uuid.uuid4().hex}")
            )
            shutil.unpack_archive(db_path, extract_dir=new_db_path,
                                  format = "zip")
            self.db_path = new_db_path.glob("*.duckdb").__next__()
            self.suffix = pathlib.Path(self.db_path).suffix
        else:
            if self.suffix == ".duckdb":
                # copy the duckdb file and it's wal to a new file
                new_filename = (
                    f"{pathlib.Path(db_path).stem}_{int(time.time())}{self.suffix}"
                )
                new_path = pathlib.Path(db_path).parent / new_filename
                shutil.copy2(db_path, new_path)
                self.db_path = new_path
            else:
                self.db_path = db_path

        self.data_quoted = quote(db_path)

    def read_table(
        self, table: str, *, lazy: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        query = f"SELECT * FROM {table}"

        if self.suffix == ".sqlite":
            df = pl.read_database_uri(query, uri=f"sqlite://{self.data_quoted}")
        elif self.suffix == ".duckdb":
            conn = duckdb.connect(str(self.db_path))
            df = conn.sql(query).pl()
            conn.close()
        else:
            raise ValueError("Only .sqlite and .duckdb files are supported")
        return df.lazy() if lazy else df


class FeatureInfo(object):
    def __init__(
        self,
        data_reference: pl.DataFrame,
        time_reference: Optional[pl.DataFrame] = None,
    ):
        self.data_reference = data_reference
        self.time_reference = time_reference
        self.max_time_id = None

    def get_vocabulary_size(self) -> int:
        return int(self.data_reference.select("columnId").max().item())

    def get_numerical_feature_ids(self) -> torch.Tensor:
        return (
            self.data_reference.filter(pl.col("isBinary") == "N")
            .select("columnId")
            .sort("columnId")
            .to_torch()
            .squeeze(1)
        )

    def get_max_time_id(self):
        return self.max_time_id

    def set_max_time_id(self, max_time_id: int):
        self.max_time_id = max_time_id


def read_data(
    path: str, lazy: bool = True, data_reference: Optional[pl.DataFrame] = None
) -> dict[str, pl.DataFrame | pl.LazyFrame]:
    """
    Reads the data from the given path and returns a dict of
    lazy polars dataframes.
    """
    reader = DBReader(path)
    analysis_ref = reader.read_table("analysisRef", lazy=lazy)
    if data_reference is None:
        data_ref = reader.read_table("covariateRef", lazy=lazy)
    else:
        data_ref = data_reference.lazy()
    data_ref = (data_ref.
        with_columns(pl.col("columnId").cast(pl.Int32)).
        join(analysis_ref, on="analysisId").select(
        pl.all().exclude("collisions")
    ))
    data = reader.read_table("covariates", lazy=lazy)
    # check if there is a timeId column in data
    time = "timeId" in data.collect_schema().names()
    if time:
        time_ref = reader.read_table("timeRef", lazy=lazy)
    else:
        time_ref = None

    return {"data": data, "data_ref": data_ref, "time_ref": time_ref if time else None}


def fill_missing_means_zero(
    data: pl.LazyFrame,
    data_ref: pl.LazyFrame,
    with_time: bool = False,
) -> pl.LazyFrame:
    """
    Fills the missing values in the data with 0.0 for features that have
    missingMeansZero set to 'Y' in the data reference and missing values.
    """
    missing_means_zero = data_ref.filter(pl.col("missingMeansZero") == "Y").select(
        "columnId"
    )
    columns = ["rowId", "columnId", "covariateValue"]
    if with_time:
        data = data.with_columns(pl.col("timeId").cast(pl.Int32))
        columns.append("timeId")
    data = data.select(columns).with_columns(
        pl.col("rowId") - 1  # Convert R rowId to Python rowId (0-indexed)
    )
    missing_means_zero_columns = missing_means_zero.collect()["columnId"].to_list()
    all_rows = data.select("rowId").unique()
    all_pairs = all_rows.join(missing_means_zero, how="cross")
    zero_filled = (
        data.filter(pl.col("columnId").is_in(missing_means_zero_columns))
        .join(all_pairs, on=["rowId", "columnId"], how="right")
        .with_columns(pl.col("covariateValue").fill_null(0.0))
        .select(columns)
    )
    remaining = data.filter(pl.col("columnId").is_in(missing_means_zero_columns).not_())
    sort_columns = ["rowId", "columnId"]
    if with_time:
        sort_columns.append("timeId")
    return pl.concat([zero_filled, remaining]).sort(sort_columns)


def aggregate_sequences(
    data: pl.LazyFrame,
    with_time: bool = False,
) -> pl.LazyFrame:
    """
    Groups the data by rowId and creates lists of feature_ids, feature_values,
    and optionally time_ids.
    """
    aggs = [
        (
            pl.col("columnId").sort_by(["timeId", "columnId"]) if with_time else pl.col("columnId")
        ).alias("feature_ids"),
        (
            pl.col("covariateValue").sort_by("timeId")
            if with_time
            else pl.col("covariateValue")
        ).alias("feature_values"),
    ]
    if with_time:
        data = data.with_columns(pl.col("timeId")
                                 .fill_null(0)
                                 .cast(pl.Int32))
        aggs.append(pl.col("timeId").sort_by("timeId").alias("time_ids"))
    grouped = data.group_by("rowId").agg(*aggs)
    all_rows = (
        data.select("rowId").unique().with_columns((pl.col("rowId")).alias("rowId"))
    )
    return all_rows.join(grouped, on="rowId", how="left").with_columns(
        pl.col("feature_ids").fill_null(pl.lit([0], dtype=pl.List(pl.Int32))),
        pl.col("feature_values").fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
        *(
            [pl.col("time_ids").fill_null(pl.lit([0], dtype=pl.List(pl.Int32)))]
            if with_time
            else []
        ),
    )


def prefix_and_pad_expr(
    expr: pl.Expr,
    max_sequence_length: int,
    pad_value: float = 0.0,
    prefix: Optional[Sequence] = None,
) -> pl.Expr:
    """
    Prefix and pad
    """
    padded = pad_truncate_expr(
        expr, max_sequence_length=max_sequence_length, pad_value=pad_value
    )
    if prefix is None:
        return padded

    total_length = len(prefix) + max_sequence_length
    padded = pl.concat_list(pl.lit(prefix), padded).list.slice(0, total_length)
    return padded


def pad_truncate_expr(
    expr: pl.Expr, max_sequence_length: int, pad_value: Union[float, int] = 0
) -> pl.Expr:
    """
    Given a List-typed Expr, extend it with "pad_value" up to "max_sequence_length"
    and then slice it to that length.
    """
    return expr.list.eval(
        pl.element()
        .extend_constant(pad_value, max_sequence_length)
        .slice(0, max_sequence_length)
    )


def pad_prefix_all(
    lf: pl.LazyFrame,
    *,
    max_sequence_length: int,
    truncation: str = "tail",
    use_time: bool = False,
) -> pl.LazyFrame:
    if truncation != "tail":
        raise NotImplementedError("Only 'tail' truncation is implemented")

    rules = {
        "feature_ids": dict(pad_value=0, prefix=None),
        "feature_values": dict(pad_value=0.0, prefix=None),
    }
    if use_time:
        rules["time_ids"] = dict(pad_value=0, prefix=[0])

    exprs = [
        prefix_and_pad_expr(
            pl.col(col),
            max_sequence_length=max_sequence_length,
            **params,
        ).alias(col)
        for col, params in rules.items()
    ]
    return lf.with_columns(exprs)


def insert_time_tokens(
    data: pl.LazyFrame,
    data_ref: pl.LazyFrame,
    strategy: str,
) -> tuple[LazyFrame, LazyFrame]:
    if strategy not in ["coarse", "fine"]:
        raise ValueError(
            "Invalid strategy for time tokens. Use 'coarse' or 'fine'."
            "You provided: " + strategy
        )

    offset = get_offset(data_ref)
    data = (
        data
        .sort(["rowId", "timeId"])
        .with_columns(
           [
                pl.cum_count("rowId").cast(pl.Float64).alias("_pos"),  # 0,1,2…
                (
                        pl.col("timeId")  # δt (days)
                        - pl.col("timeId").shift(1).over("rowId")
                )
                .fill_null(0)
                .alias("_delta"),
            ]
        )
    )
    delta = pl.col("_delta")

    token_number_expr = (
        pl.when(delta < 28)
        .then(delta // 7)  # W0-W3  → 0-3
        .when(delta < 365)
        .then(4 + ((delta - 28) // 30))  # M1-M11 → 5-15
        .otherwise(pl.lit(15))  # LT     → 15
    ).cast(pl.Int32)

    token_feature_id_expr = pl.lit(offset) + token_number_expr

    # create time tokens
    tokens = (
        data.filter(delta > 0)  # no token before first visit
        .with_columns(
            [
                token_feature_id_expr.alias("columnId"),
                pl.lit(0.0).alias("covariateValue"),  # ← your call, keep or change
                (pl.col("_pos") - 0.5).alias("_pos"),
                # ensures token sorts *before*
            ]
        )
        .select("rowId", "columnId", "covariateValue", "timeId", "_pos")
    )
    final_columns = ["rowId", "columnId", "covariateValue", "timeId", "_pos"]
    data_aug = (
        pl.concat([data.select(final_columns), tokens.select(final_columns)])
        .sort(["rowId", "_pos"])
        .drop("_pos")
    )
    data_ref_aug = augment_data_reference(data_ref, offset=offset, strategy=strategy)
    return data_aug, data_ref_aug


def augment_data_reference(data_ref, strategy, offset) -> pl.LazyFrame:
    if strategy == "coarse":
        token_str = (
            ["W0", "W1", "W2", "W3"]  # 0‥3
            + [f"M{i}" for i in range(1, 12)]  # 4‥14
            + ["LT"]  # 15
        )
    else:
        raise NotImplementedError(
            "Only 'coarse' strategy is implemented for time tokens"
        )

    extra_rows = pl.DataFrame(
        {
            "columnId": [offset + i for i in range(len(token_str))],
            "covariateName": token_str,
        },
        schema_overrides={"columnId": pl.Int32, "covariateName": pl.Utf8},
    )
    data_ref_schema = data_ref.collect_schema()
    extra_lazy = (
        extra_rows.lazy()
        .with_columns(
            [
                *(
                    pl.lit(None).cast(dtype).alias(col)
                    for col, dtype in data_ref_schema.items()
                    if col not in extra_rows.columns
                )
            ]
        )
        .select(data_ref_schema.keys())  # same order
    )
    data_ref_aug = pl.concat([data_ref, extra_lazy])
    return data_ref_aug


def convert_to_long(data: pl.LazyFrame) -> pl.LazyFrame:
    events = (
        data.select(
            "rowId",
            pl.col("feature_ids").alias("f_ids"),
            pl.col("feature_values").alias("f_vals"),
            pl.col("time_ids").alias("t_ids"),
        )
        .explode(["f_ids", "f_vals", "t_ids"])
        .rename(
            {
                "f_ids": "feature_id",
                "f_vals": "feature_value",
                "t_ids": "time_ids",
            }
        )
        .sort(["rowId", "time_ids"])
        .with_columns(
            [
                pl.cum_count("rowId").cast(pl.float64).alias("_pos"),  # 0,1,2…
                (
                    pl.col("time_ids")  # δt (days)
                    - pl.col("time_ids").shift(1).over("rowid")
                )
                .fill_null(0)
                .alias("_delta"),
            ]
        )
    )
    return events


def get_offset(data_ref: pl.LazyFrame) -> int:
    """
    Get the offset for the time tokens based on the maximum columnId in the data reference.
    """
    return data_ref.select(pl.col("columnId")).max().collect().item() + 1

