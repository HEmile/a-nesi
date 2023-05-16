"""
code adapted from deepproblog repo
"""
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from torch.utils.data import random_split


_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

_full_train_set = torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    )
_train_set, _val_set = random_split(_full_train_set, [50000, 10000])


datasets = {
    "train": _train_set,
    "val": _val_set,
    "full_train": _full_train_set,
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number


def addition(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
    )

def multiplication(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""

    return MNISTOperator(
        dataset_name=dataset,
        function_name="multiplication",
        operator=math.prod,
        size=n,
        arity=2,
        seed=seed,
    )


class MNISTOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[np.array, np.array, int, List[int]]:
        l1, l2 = self.data[index]
        label, digits = self._get_label(index)
        l1 = torch.stack([self.dataset[x][0][0] for x in l1])
        l2 = torch.stack([self.dataset[x][0][0] for x in l2])
        return l1, l2, label, digits

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        digits = [[self.dataset[j][1] for j in i] for i in mnist_indices]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            digits_to_number(di) for di in digits
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result, digits

    def __len__(self):
        return len(self.data)
