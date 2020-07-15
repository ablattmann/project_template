# General Project Template #

This template can be used to run deep learning projects using [`pytorch`](https://pytorch.org/) as a framework and [`wandb`](https://www.wandb.com/) for logging.

Training: From the **root of this repo**, run:

`python main.py --config "config/<config_name>" --gpu <gpu_ids, comma separated> --mode train --restart <True/False>`

Testing: Run 
`python main.py --config "config/<config_name>" --gpu <gpu_ids, comma separated> --mode train`


## Requirements ##

```
torch
torchvision
wandb
yaml
```

## Directory structure ##

### config ###

Here goes all the the config information. Store them in a `yaml`-file, which contains at least the subdicts `general` and `data` (see `config/test_config.yaml` for an examplary file).

### data ###

Here are all the different datasets and helper methods for data preprocessing. All datasets should inherit from the `BaseDataset`-class from `data/base_dataset.py`.

### experiments ###

Here are all the different experiments, which should inherit from the `Experiment`-class from `experiments/experiment.py`.

### utils ###

Helper functions, metrics etc.

### models ###

Network-architecures which are trained within the different experiments and their respective modules. 