import os
import numpy as np
import random
import torch
import wandb
import progressbar
import random

from utils import *

def get_dataset_stats(dataset, max_samples=None):
    len_ds = len(dataset)
    n_targets = len(set(dataset.targets))

    if (max_samples is None) or (max_samples > len_ds): max_samples = len_ds
  
    sample, target = dataset[0]
    sample_cat = sample.view(sample.shape[0], -1)

    with progressbar.ProgressBar(max_value=max_samples) as bar:
        for i in range(0, max_samples - 1):
            idx = random.randint(1, len_ds)
            sample, target = dataset[idx]
            print(sample.shape)
            sample_cat = torch.cat((sample_cat, sample.view(sample.shape[0], -1)), dim=1)
            bar.update(i)

    return torch.mean(sample_cat, dim=1), torch.std(sample_cat, dim=1)


def get_dataset_stats_by_class(dataset, max_samples=None):

    return None


def visualise_random_batch(dataset, n=5):
    
    return None


if __name__ == "__main__":
    # Some placeholder hyperparameters for set_up_train
    parameters = dict(
        nepochs=100,
        
        data_name="xray-data",                                                 
        target_code={"covid":0, "lung_opacity":1, "pneumonia":2, "normal":3}, 
        input_height=299,
        input_width=299,
        input_channels=1,
        
        batch_size=64,
        test_batch_size=64,
        learning_rate=1e-1,
        momentum=0.0,
        
        model_name="SimpleConvNet",                                          
        dataset="XRAYTensorDataset",                                           
        criterion_name="CrossEntropyLoss",
        optimizer_name="SGD",
        
        valid_split=0.2,
        train_transform=train_transform,
        test_transform=test_transform,
        device=set_device("cuda"),
        )


    config = setup_config_offline(parameters)

    # Set seed and devices
    set_seed(42)

    # make dataset
    trainds, validds = make_dataset(config=config, train=True)

    mean, std = get_dataset_stats(trainds, max_samples=None)
    print(mean, std)

