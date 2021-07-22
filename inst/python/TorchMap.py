import torch


def map_python(datas, maxCol, maxRow, maxT=None, matrix=None):
    if maxT is not None:
        indexes = datas[:, 0:3] - 1
        matrixt = torch.sparse.FloatTensor(torch.LongTensor(indexes.T), torch.FloatTensor(datas[:, 3]),
                                           torch.Size([maxRow, maxCol, maxT]))
    else:
        indexes = datas[:, 0:2] - 1
        matrixt = torch.sparse.FloatTensor(torch.LongTensor(indexes.T), torch.FloatTensor(datas[:, 2]),
                                           torch.Size([maxRow, maxCol]))
    if matrix is None:
        return matrix
    else:
        matrix = matrix.add(matrixt)
    return matrix
