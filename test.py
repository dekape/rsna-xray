import os
import sys
import numpy as np

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, ToPILImage

from networks import *
from datasets import *
from utils import *

 
   
if __name__ == "__main__":
    # Set test transforms for normalisation 
    test_transform = Compose([
        Normalize(mean=[0.5064], std=[0.2493])
    ])
    

    # Set training static parameters and hyperparameters
    parameters = dict(
        data_name="xray-data",                                                     # will search folder ./{data_name}/train and ./{data_name}/test
        target_code={"covid":0, "lung_opacity":1, "pneumonia":2, "normal":3},      # each key should be a subfolder with the training data ./{data_name}/train/{key}
        input_height=299,
        input_width=299,
        input_channels=1,
        
        test_batch_size=64,
        
        model_name="ResNet18Wrapper",                                      # will search model in networks.py, must be a subclass of nn.Module and take as input param "num_classes" referring to the number of classes for classification
        dataset="XRAYTensorDataset",                                            # will search dataset in datasets.py, must be a subclass of torch.utils.data.Dataset, inputs must be img_paths, targets and transform, output of __getitem__ must be an image sample (C, H, W) and its target
        
        test_transform=test_transform,
        device=set_device("cuda"),
        )

    # set up offline wandb config
    config = setup_config_offline(parameters)

    # Set seed
    set_seed(42)
    
    # Set up evaluation with parameters, and load trained model
    # model_name = "swapped_bright_vortex"
    model_name = "bright-vortex-61_epoch90"
    # model_name = "poisson100"
    path_to_trained_model = os.path.join(".", "wandb", "run-20210423_165728-2q4iavg1", "files", model_name+".pth")
    model, test_ds  = set_up_test(config, path_to_model=path_to_trained_model, test_transform=parameters["test_transform"])
    
    print(model)
    
    # # Get Predictions
    # print("\n Predicting...") 
    # img_paths, y_preds, y_probs = predict(model=model, dataset=test_ds, config=config)
    # print(y_probs)

    # # Save predictions to csv file
    # print("\n Saving prediction file...")
    # split_char = "\\" if sys.platform=="win32" else "/"
    # img_names = [pth.split(split_char)[-1].split(".")[0] for pth in img_paths]
    # save_preds_csv(img_names, y_preds, filename="prediction_"+model_name+".csv")
    # # save_probs_csv(img_names, y_probs, target_names=list(config.target_code.keys()), filename="probabilities.csv")
    # print("\n DONE! \n")
    

    # # ! STUDENTS WON'T HAVE ACCESS TO THE test_key.csv FILE
    # key_filepath = os.path.join("submission", "test_key.csv")
    # pred_filepath = os.path.join("submission", model_name+".csv")
    # acc, auc, f1, cm, keydf, testdf = get_pred_metrics(key_filepath, pred_filepath)
        
    
    # # print(cm)
    # print("\n\n F1 SCORE: ", f1)
    # print("\n\n ACCURACY : ", acc)
    # print("\n\n AUC: ", auc)

        
      
    