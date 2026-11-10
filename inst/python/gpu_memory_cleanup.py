# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import gc

import torch


def memory_cleanup():
    """Free memory that nothing refers to any more, after running out of memory.

    Tensors that are still referenced are left alone. Training continues with
    the next hyperparameter combination in the same process, so freeing their
    storage would break whatever still uses them.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
