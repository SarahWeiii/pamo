from typing import Union
import numpy as np
import torch
import logging
import sys

import warp as wp
from warp.types import type_size_in_bytes


mat32 = wp.types.matrix(shape=(3, 2), dtype=wp.float32)
mat22 = wp.types.matrix(shape=(2, 2), dtype=wp.float32)


def wp_slice(a: wp.array, start, end):
    """Utility function to slice a warp array along the first dimension
    """

    assert a.is_contiguous
    assert 0 <= start <= end <= a.shape[0]
    return wp.array(
        ptr=a.ptr + start * a.strides[0],
        dtype=a.dtype,
        shape=(end - start, *a.shape[1:]),
        strides=a.strides,
        device=a.device,
        copy=False,
        owner=False,
    )
    

def convert_to_wp_array(
    a: Union[np.ndarray, torch.Tensor, wp.array, list], dtype=None, device=None
):
    """Utility function to convert numpy or torch tensor to warp array
    """
    if isinstance(a, np.ndarray):
        return wp.from_numpy(a, dtype=dtype, device=device)
    elif isinstance(a, torch.Tensor):
        return wp.from_torch(a, dtype=dtype).to(device)
    elif isinstance(a, list):
        return wp.from_numpy(np.array(a), dtype=dtype, device=device)
    elif isinstance(a, wp.array):
        assert dtype is None or dtype == a.dtype
        assert device is None or device == a.device
        return a
    else:
        raise ValueError(f"unsupported type {type(a)}")


class ConsoleFormatter(logging.Formatter):
    """ https://stackoverflow.com/a/56944256 """
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: blue + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


console_formatter = ConsoleFormatter()
file_formatter = logging.Formatter(fmt="%(levelname)s - %(message)s (%(filename)s:%(lineno)d)")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

stage3_logger = logging.getLogger('sapienipc')
stage3_logger.setLevel(logging.INFO)
stage3_logger.handlers.clear()
stage3_logger.propagate = False
stage3_logger.addHandler(console_handler)

