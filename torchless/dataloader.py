import math
import numpy as np
from typing import List, Any, Iterator

from config import dataloader_settings

from dataset import TidigitDataset, collate_fn


class DataLoader:
    def __init__(
        self,
        dataset: TidigitDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        collate_fn: Any = collate_fn,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        # initialize the indices for shuffling
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)

        # current reading position
        self.current_idx = 0

    def __iter__(self) -> Iterator:
        self.currend_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Any:
        start = self.current_idx
        end = self.current_idx + self.batch_size

        # if the sides of the batch are out of the dataset, raise StopIteration
        if start >= len(self.dataset):
            raise StopIteration

        if end > len(self.dataset):
            if self.drop_last:
                raise StopIteration
            end = len(self.dataset)

        # get the indeces of current batch
        batch_indices = self.indices[start:end]

        # extract the abtch from the dataset
        batch = [self.dataset[i] for i in batch_indices]

        # if a collate_fn is provided, then use it
        if self.collate_fn:
            batch = self, self.collate_fn(batch)

        self.current_idx = end

        return batch

    def __len__(self) -> int:
        """return num of current batch"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)
