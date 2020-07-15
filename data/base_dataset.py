import torch
from torch.utils.data import Dataset
import numpy as np
from abc import abstractmethod


class BaseDataset(Dataset):
    def __init__(
        self,
        transforms,
        datakeys:list,
        **kwargs,
    ):
        super().__init__()

        # list of keys for the data that shall be retained
        self.datakeys = datakeys
        # torchvision.transforms
        self.transforms = transforms
        # key:value mappings for every datakey in self.datakeys
        self._output_dict = {"random_numbers":self._get_random_nrs}

        # the data that's held by the dataset
        self.datadict = {}

    def __getitem__(self, idx):
        # collect outputs
        data = {
            key: self._output_dict[key](idx)
            for key in self.datakeys
        }
        return data

    # only as an example
    def _get_random_nrs(self,idx):
        return np.random.randint(0,1000)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_test_app_images(self) -> dict:
        pass
