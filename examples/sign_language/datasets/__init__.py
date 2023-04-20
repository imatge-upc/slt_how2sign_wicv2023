# This code is a modification of the faiseq task registration modules,
# which are licensed under the MIT license found in the LICENSE file
# in the root directory.

import os
import importlib

from torch.utils.data import Dataset

from .base import SignLanguageDataset

DATASET_REGISTRY = {}
DATASET_CLASS_NAMES = set()

def register_dataset(name):
    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Cannot register duplicate dataset ({name})")
        if not issubclass(cls, Dataset):
            raise ValueError(
                f"Task ({name}: {cls.__name__}) must extend FairseqTask"
            )
        if cls.__name__ in DATASET_CLASS_NAMES:
            raise ValueError(
                f"Cannot register dataset with duplicate class name ({cls.__name__})"
            )
        DATASET_REGISTRY[name] = cls
        DATASET_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_dataset_cls


def get_dataset(name):
    return DATASET_REGISTRY[name]

def import_datasets(datasets_dir):
    for file in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module('.' + dataset_name, package=__name__)

datasets_dir = os.path.dirname(__file__)
import_datasets(datasets_dir)
