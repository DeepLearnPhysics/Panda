"""
General utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import inspect
from typing import Callable, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate

from .logging import get_logger
logger = get_logger(__name__)


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def filter_kwargs(
    func: Callable,
    kwargs: Dict[str, Any],
    strict: bool = False,
    warn: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Filter kwargs to only include valid parameters for a function/class.
    
    Args:
        func: function or class to filter kwargs for
        kwargs: dict of kwargs to filter
        strict: if True, raise TypeError on unknown keys
        warn: if True, print warning about ignored keys
        
    Returns:
        filtered: dict with only valid kwargs
        ignored: list of ignored keys
    """
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    
    # check if function accepts **kwargs
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    
    if has_var_keyword:
        return kwargs.copy(), []
    
    filtered = {}
    ignored = []
    for k, v in kwargs.items():
        if k in valid_params:
            filtered[k] = v
        else:
            ignored.append(k)
    
    if ignored:
        if strict:
            raise TypeError(f"Unknown kwargs for {func.__name__}: {ignored}")
        elif warn:
            logger.info(f"{func.__name__}: Ignoring unknown kwargs: {ignored}")
    
    return filtered, ignored



def collate_fn(batch, mix_prob=0):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)
