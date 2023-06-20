import time
import pathlib

import polars as pl
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self,
                 data,
                 labels=None,
                 numerical_features=None):
        """
        data: path to a covariates dataframe either arrow dataset or sqlite object
        labels: a list of either 0 or 1, 1 if the patient got the outcome
        numerical_features: list of indices where the numerical features are
        """
        start = time.time()
        data = pl.scan_ipc(pathlib.Path(data).joinpath('covariates/*.arrow'))
        observations = data.select(pl.col('rowId').max()).collect()[0, 0]
        # detect features are numeric
        if not numerical_features:
            self.numerical_features = data.groupby(by='columnId') \
                .n_unique().filter(pl.col('covariateValue') > 1).select('columnId').collect()['columnId']
        else:
            self.numerical_features = pl.Series('num', numerical_features)

        if labels:
            self.target = torch.as_tensor(labels)
        else:
            self.target = torch.zeros(size=(observations,))

        # filter by categorical columns,
        # sort and group_by columnId
        # create newColumnId from 1 (or zero?) until # catColumns
        # select rowId and newColumnId
        # rename newColumnId to columnId and sort by it
        data_cat = data.filter(~pl.col('columnId')
                               .is_in(self.numerical_features)) \
            .sort(by='columnId').with_row_count('newColumnId'). \
            with_columns(pl.col('newColumnId').first().over('columnId')
                         .rank(method="dense")) \
            .select(pl.col('rowId'), pl.col('newColumnId').alias('columnId')).sort('rowId') \
            .with_columns(pl.col('rowId') - 1).collect()
        cat_tensor = torch.as_tensor(data_cat.to_numpy())
        tensor_list = torch.split(cat_tensor[:, 1], torch.unique_consecutive(cat_tensor[:, 0], return_counts=True)[1].
                                  tolist())

        # because of subjects without cat features, I need to create a list with all zeroes and then insert
        # my tensorList. That way I can still index the dataset correctly.
        total_list = [torch.as_tensor((0,))] * observations
        idx = data_cat['rowId'].unique().to_list()
        for i, i2 in enumerate(idx):
            total_list[i2] = tensor_list[i]
        self.cat = torch.nn.utils.rnn.pad_sequence(total_list, batch_first=True)
        self.cat_features = data_cat['columnId'].unique()

        # numerical data,
        # N x C, dense matrix with values for N patients/visits for C numerical features
        if pl.count(self.numerical_features) == 0:
            self.num = None
        else:
            numerical_data = data.filter(pl.col('columnId').is_in(self.numerical_features)). \
                with_row_count('newColumnId').with_columns(pl.col('newColumnId').first().over('columnId').
                                                           rank(method="dense") - 1, pl.col('rowId') - 1) \
                .select(pl.col('rowId'), pl.col('newColumnId').alias('columnId'), pl.col('covariateValue')).collect()
            indices = torch.as_tensor(numerical_data.select(['rowId', 'columnId']).to_numpy(), dtype=torch.long)
            values = torch.as_tensor(numerical_data.select('covariateValue').to_numpy(), dtype=torch.float)
            self.num = torch.sparse_coo_tensor(indices=indices.T,
                                               values=values.squeeze(),
                                               size=(observations, pl.count(self.numerical_features))).to_dense()
        delta = time.time() - start
        print(f'Processed data in {delta:.2f} seconds')

    def get_numerical_features(self):
        return self.numerical_features

    def get_cat_features(self):
        return self.cat_features

    def __len__(self):
        return self.target.size()[0]

    def __getitem__(self, item):
        if self.num is not None:
            batch = {"cat": self.cat[item, :].squeeze(),
                     "num": self.num[item, :].squeeze()}
        else:
            batch = {"cat": self.cat[item, :].squeeze(),
                     "num": None}
        if batch["cat"].dim() == 1:
            batch["cat"] = batch["cat"].unsqueeze(0)
        if batch["num"] is not None and batch["num"].dim() == 1:
            batch["num"] = batch["num"].unsqueeze(0)
        return [batch, self.target[item].squeeze()]