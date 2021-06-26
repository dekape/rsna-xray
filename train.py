import os
import numpy as np
import random
import wandb

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, ToPILImage, ColorJitter, Resize, RandomCrop, RandomHorizontalFlip

from networks import *
from datasets import *
from utils import *


   
if __name__ == "__main__":
    # Set training and testing/validate transforms for normalisation and data augmentation
    train_transform = Compose([
        ToPILImage(),
        Resize(310),
        RandomCrop(299),
        RandomRotation(10, fill=(0,)),
        RandomHorizontalFlip(p=0.3),
        ColorJitter(brightness=0.5, contrast=0.5),
        ToTensor(),
        Normalize(mean=[0.5064], std=[0.2493]), 
    ])

    test_transform = Compose([
        Normalize(mean=[0.5064], std=[0.2493])
    ])
    

    # Set training static parameters and hyperparameters
    parameters = dict(
        nepochs=100,
        
        data_name="xray-data",                                                 # will search folder ./{data_name}/train and ./{data_name}/test
        target_code={"covid":0, "lung_opacity":1, "pneumonia":2, "normal":3},  # each key should be a subfolder with the training data ./{data_name}/train/{key}
        input_height=299,
        input_width=299,
        input_channels=1,
        
        batch_size=64,
        test_batch_size=64,
        learning_rate=1e-1,
        momentum=0.5,
        
        model_name="ResNet18Wrapper",                                           # will search model in networks.py, must be a subclass of nn.Module and take as input param "num_classes" referring to the number of classes for classification
        dataset="XRAYTensorDataset",                                            # will search dataset in datasets.py, must be a subclass of torch.utils.data.Dataset, inputs must be img_paths, targets and transform, output of __getitem__ must be an image sample (C, H, W) and its target
        criterion_name="CrossEntropyLoss",
        optimizer_name="SGD",
        
        valid_split=0.25,
        balance_loader=True,
        train_transform=train_transform,
        test_transform=test_transform,
        
        device=set_device("cuda"),
        )
  
    
    # tell wandb to get started
    with wandb.init(project="rsna", config=parameters, entity="dekape") as wb:  
        # Model saving settings
        save_model = True
        save_frequency = 5
        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Set seed
        set_seed(42)
        
        # Set up run with parameters
        model, criterion, optimizer, train_loader, valid_loader = set_up_train(config, 
            train_transform=parameters["train_transform"], valid_transform=parameters["test_transform"])

        # # Let wandb watch the model and the criterion
        wandb.watch(model, criterion)
        
        # Training loop
        print("\n\nTraining started ...")
        for epoch in range(config.nepochs):
            train_loss, train_accuracy, train_auc = train(model, optimizer, criterion, train_loader, config)
            validation_loss, validation_accuracy, validation_auc = validate(model, criterion, valid_loader, config)
            
            log = {"epoch": epoch, 
                "train_loss":train_loss.item(), "train_accuracy": train_accuracy.item(), "train_auc": train_auc.item(), 
                "valid_loss":validation_loss.item(), "valid_accuracy":validation_accuracy.item(), "validation_auc":validation_auc.item()}
            wandb.log(log)
            print(log)

            # Saving model
            if save_model and (epoch % save_frequency == 0 or epoch==config.nepochs-1):
                model_dir =  os.path.join(wb.dir, "%s_epoch%g.pth"%(wb.name, epoch))
                print("\n Saving model at %s \n"%(model_dir))
                torch.save(model.state_dict(), model_dir)

        
        
        
        
      
    