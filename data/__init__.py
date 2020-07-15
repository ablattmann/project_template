from data.base_dataset import BaseDataset
from torchvision import transforms as tt

# add key value pair for datasets here, all datasets should inherit from base_dataset
__datasets__ = {"Base":BaseDataset}


# returns only the class, not yet an instance
def get_transforms(config):
    return {"Base":tt.ToTensor()}


def get_dataset(config,custom_transforms=None):
    dataset = __datasets__[config["dataset"]]
    if custom_transforms is not None:
        print("Returning dataset with custom transform")
        transforms = custom_transforms
    else:
        transforms = get_transforms(config)[config["dataset"]]
    return dataset, transforms
