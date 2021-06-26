
import os
import random
import numpy as np
import torch
import progressbar
import pandas as pd
import random
import wandb

from pycm import *

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torch.utils.data import Dataset 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

import datasets
import networks


EXTENSIONS = ["PNG", "png"]

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True

    return True


def set_device(device="cpu"):
    if device != "cpu":
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print("Cuda installed! Running on GPU %s!"%torch.cuda.get_device_name())
            device="cuda:0"
        else:
            device="cpu"
            print("No GPU available! Running on CPU")
    return device
    

def is_valid_image(img_path):
    ext = img_path.split(".")[-1]
    if ext in EXTENSIONS:
        return True
    else:
        return False


def get_params_to_update(model):
    """ Returns list of model parameters that have required_grad=True"""
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


def print_single_stats(key, val):
    print(" %-45s %-15s %15s"%(key, ":", val))
    return None


def setup_config_offline(parameters):
    """
    Sets up wandb.config for pure offline runs, bypassing wanb.init 
    """
    config = wandb.sdk.wandb_config.Config()
    for key in parameters:
        config.__setitem__(key, parameters[key])
    return config
  
   
def save_preds_csv(img_names, predictions, filename="prediction.csv"):
    """ Saves predictions into csv file formated as
    name,target
    name1,1
    name2,3
    ...
    name948,2
    name949,2
    """
    assert(len(img_names) == len(predictions))
    
    df = pd.DataFrame(columns=["name", "target"])
    for i in range(len(img_names)):
        df.loc[i] = [img_names[i], predictions[i]]
        
    df.to_csv(os.path.join(".", filename), index=False)
    return None    


def save_probs_csv(img_names, probabilities, target_names, filename="probabilities.csv"):
    """ Saves probabilities into csv file formated as
    name,target_name0, target_name1, etc..
    name1,1.20066126e-08, 8.89000356e-01, etc ...
    name2,4.66548689e-10, 1.10999264e-01, etc ...
    ...
    name948,1.17128581e-11, 7.14919167e-07 etc ...
    name949,2 5.63821038e-11, 1.00000000e+00 etc ...
    """
    assert(len(img_names) == len(probabilities))
    
    df = pd.DataFrame(columns=["name"]+target_names)
    for i in range(len(img_names)):
        df.loc[i] = [img_names[i]] + list(probabilities[i])
    df.to_csv(os.path.join(".", filename), index=False)
    return None  


def get_pred_metrics(key_filepath, pred_filepath, average="macro"):
    """ Gets F1 value out a key and predicted csv files
    Files must be formated as
    name,target
    name1,1
    name2,3
    ...
    name948,2
    name949,2
    """
    key = pd.read_csv(key_filepath)
    pred = pd.read_csv(pred_filepath)
    assert len(key.index) == len(pred.index)

    merged = pd.merge(key, pred, on=['name'], how='inner')
    y_true, y_pred = merged["target_x"].to_numpy(), merged["target_y"].to_numpy()
    
    cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    f1 = f1_score(list(y_true), list(y_pred), average=average)
    auc = roc_auc_score_multiclass(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    
    return acc, auc, f1, cm, key, pred
    
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    total_no_imgs = len(actual_class)
    auc = 0
    ratio = 0
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        # print(new_actual_class)
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average, multi_class="ovr")
        roc_auc_dict[per_class] = roc_auc
        
        ratio_per_class = sum(actual_class == per_class) / total_no_imgs
        ratio += ratio_per_class
        auc += roc_auc * ratio_per_class
        print(ratio_per_class)
    
    assert ratio == 1 
    print(roc_auc_dict)
    # auc = sum(list(roc_auc_dict.values())) / len(unique_class)
    return auc

    
def get_img_paths(config, train):
    """
    Gets image paths for training and testing
    If train=True, uses keys of config.train_code to look for subroots
    | | | ./
    | | | | |{config.data_name}/
    | | | | | | | train/
    | | | | | | | | | {config.target_code}/
    | | | | | | | | | | | *.png or *.PNG images
    
    if train=False, loads absolute paths for all images in the folder structure
    | | | ./
    | | | | |{config.data_name}/
    | | | | | | | test/
    | | | | | | | | | *.png or *.PNG images
    
    """
    root_path = os.path.join(".", config.data_name)
    load_path = os.path.join(root_path, "train") if train else os.path.join(root_path, "test")
    print("Reading image paths in %s"%load_path)
    
    img_paths = []
    targets = []
    
    if train:
        target_code = config.target_code
        for category in target_code:
            for root, _, fnames in sorted(os.walk(os.path.join(load_path, category))):
                for fname in fnames:
                    if is_valid_image(fname):
                        img_paths.append(os.path.join(root, fname))
                        targets.append(target_code[category])
    else:
        for root, _, fnames in sorted(os.walk(load_path)):
            for fname in fnames:
                if is_valid_image(fname):
                    path = os.path.join(root, fname)
                    img_paths.append(path)
                    targets.append(0) # targets list for test is only a placeholder

    return img_paths, targets


def make_dataset(config, train, train_transform=None, test_transform=None):
    """
    Split image paths into training and validation sets, creates the dataset objects for both training and validation 
    """
    try:
        CustomTensorDataset = getattr(datasets, config.dataset)
    except:
        raise NotImplementedError("Dataset of name %s has not been found in file datasets.py "%config.dataset)     
    
    if train:
        img_paths, targets = get_img_paths(config, train=True)
        shuffler = StratifiedShuffleSplit(n_splits=1, test_size=config.valid_split, random_state=42).split(img_paths, targets)
        train_idx, valid_idx = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]

        X_train = [img_paths[i] for i in train_idx]
        y_train = [targets[i] for i in train_idx]

        X_valid = [img_paths[i] for i in valid_idx]
        y_valid = [targets[i] for i in valid_idx]
        
        trainds = CustomTensorDataset(X_train, y_train, train_transform)
        validds = CustomTensorDataset(X_valid, y_valid, test_transform)
        
        return trainds, validds
    
    else:
        img_paths, targets = get_img_paths(config, train=False)
        testds = CustomTensorDataset(img_paths, targets, test_transform)
        return testds

    return None
    

def make_weights_for_balanced_classes(images, targets):
    """Adapted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703"""                        
    
    nclasses = len(set(targets))
    
    # Number of occurences of each class
    count_per_class = [0] * nclasses                                           
    for t in targets:                                                         
        count_per_class[t] += 1                                                     
    
    # Weight (reciprocal of prob) per class                                    
    N = float(len(images))                                                       
    weight_per_class = [  N / float(count) for count in count_per_class]      
    
    # Expand to target list
    weight_per_datapoint = [0] * len(images)                                              
    for i in range(len(images)):                                          
        weight_per_datapoint[i] = weight_per_class[targets[i]]                               
    return weight_per_datapoint 
    
    
def make_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2, balance_loader=True):    
    if balance_loader:
        shuffle=False # sampler is mutually exclusive with shuffle
        weights = make_weights_for_balanced_classes(dataset.img_paths, dataset.targets)
        sampler = WeightedRandomSampler(weights, len(dataset.img_paths))
    else:
        sampler=None
    
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size, 
                                        shuffle=shuffle,
                                        pin_memory=pin_memory, 
                                        num_workers=num_workers,
                                        sampler=sampler)
    return loader


def make_model(config):
    try:
        model = getattr(networks, config.model_name)
    except:
        raise NotImplementedError("Model of name %s has not been found in file networks.py "%config.model_name)
    model = model(num_classes=len(config.target_code))
    model = model.to(config.device)
    config.model = model
    return model


def make_optim(config, model):
    # Optimizer
    try:
        optimizer = getattr(torch.optim, config.optimizer_name)
        optimizer = optimizer(get_params_to_update(model), lr=config.learning_rate)
        
    except:
        raise NotImplementedError("Optimizer of name %s has not been found in torch.optim"%config.optimizer_name)
    
    # Momentum
    try:
        for g in optimizer.param_groups:
            g['momentum'] = config.momentum
    except:
        config.momentum = 0
        pass

    config.optimizer = optimizer    
    return optimizer


def make_loss(config):
    try:
        criterion = getattr(torch.nn, config.criterion_name)
        criterion = criterion()    
    except:
        raise NotImplementedError("Criterion of name %s has not been found in torch.nn"%config.criterion_name)
    config.criterion = criterion
        
    return criterion


def set_up_train(config, train_transform=None, valid_transform=None):
    # Get data and make datasets
    trainds, validds = make_dataset(config=config, train=True, train_transform=train_transform, test_transform=valid_transform)

    # Make data loaders
    train_loader = make_dataloader(trainds, config.batch_size, config.balance_loader)
    valid_loader = make_dataloader(validds, config.test_batch_size, config.balance_loader)
    
    # Make model
    model = make_model(config)
    
    # Make optimizer
    optimizer = make_optim(config, model)
        
    # Make loss
    criterion = make_loss(config)


    #### Print summary ####
    # Config items
    for item in config.items():
        print_single_stats(item[0], item[1])
    
    # Number of train samples by category
    print_single_stats("\nTotal train samples", "    %i"%len(trainds))
    for category in config.target_code:
        n_this_cat = trainds.targets.count(config.target_code[category])
        print_single_stats("   %s"%category, "%g  (%.1f%%)"%(n_this_cat, 100*n_this_cat/len(trainds)))
    
    # Number of valid samples by category
    print_single_stats("\nTotal valid samples","    %i"%len(validds))
    for category in config.target_code:
        n_this_cat = validds.targets.count(config.target_code[category])
        print_single_stats("   %s"%category, "%g  (%.1f%%)"%(n_this_cat, 100*n_this_cat/len(validds)))

    # Check model compatibility with input size
    print("\nTesting model compatibility with input size...")
    sample_input = torch.zeros_like(trainds[0][0]).unsqueeze(0).expand(config.batch_size, -1, -1, -1).to(config.device)
    print_single_stats("Sample input shape", sample_input.shape)
    sample_output = model(sample_input)
    if isinstance(sample_output, tuple):
        sample_output = sample_output[0]
    print_single_stats("Sample output shape", sample_output.shape)

    return model, criterion, optimizer, train_loader, valid_loader 


def set_up_test(config, path_to_model=None, test_transform=None):
    # Get data and make dataset
    testds = make_dataset(config=config, train=False, test_transform=test_transform)
    
    # Make model
    model = getattr(networks, config.model_name)
    try:
        model = getattr(networks, config.model_name)
    except:
        raise NotImplementedError("Model of name %s has not been found in file networks.py "%config.model_name)
    model = model(num_classes=len(config.target_code))
    model = model.to(config.device)
    config.model = model
    
    # Load model
    if path_to_model is not None:
        print("Loading model: ", path_to_model)
        model.load_state_dict(torch.load(path_to_model))

    #### Print summary ####
    # Config items
    for item in config.items():
        print_single_stats(item[0], item[1])
    
    # Number of test samples
    print_single_stats("Total test samples", len(testds))

    # ! NOTE set_up_test returns dataset, not dataloader ! 
    return model, testds 


def train(model, optimizer, criterion, data_loader, config):
    model.train()
    train_loss, train_accuracy, train_auc = 0., 0., 0.
    with progressbar.ProgressBar(max_value=len(data_loader)) as bar:
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            # a2 = model(X.view(-1, config.input_channels, config.input_height, config.input_width)) 
            a2 = model(X)
            if isinstance(a2, tuple):
                a2 = a2[0]
            loss = criterion(a2, y)
            loss.backward()
            train_loss += loss*X.size(0)
            
            y_pred = F.log_softmax(a2, dim=1)  # log probability   
            train_accuracy += accuracy_score(y.cpu().detach().numpy(), y_pred.max(1)[1].cpu().detach().numpy())*X.size(0)
            train_auc += roc_auc_score_multiclass(y.cpu().detach().numpy(), y_pred.max(1)[1].cpu().detach().numpy())*X.size(0)

            optimizer.step()
            bar.update(i)  
        
    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.dataset), train_auc/len(data_loader.dataset)


def validate(model, criterion, data_loader, config):
    model.eval()
    validation_loss, validation_accuracy, validation_auc = 0., 0., 0.
    with progressbar.ProgressBar(max_value=len(data_loader)) as bar:
        for i, (X, y) in enumerate(data_loader):
            with torch.no_grad():
                X, y = X.to(config.device), y.to(config.device)
                # a2 = model(X.view(-1, config.input_channels, config.input_height, config.input_width)) 
                a2 = model(X)
                if isinstance(a2, tuple):
                    a2 = a2[0]
                loss = criterion(a2, y)
                validation_loss += loss*X.size(0)
                
                y_pred = F.log_softmax(a2, dim=1)  # log probability  
                validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.max(1)[1].cpu().numpy())*X.size(0)
                validation_auc += roc_auc_score_multiclass(y.cpu().detach().numpy(), y_pred.max(1)[1].cpu().detach().numpy())*X.size(0)
                
                             
                bar.update(i)
            
    return validation_loss/len(data_loader.dataset), validation_accuracy/len(data_loader.dataset), validation_auc/len(data_loader.dataset)        


def predict(model, dataset, config):
    model.eval()
    img_paths = dataset.img_paths
    y_preds, y_probs = [], []
    
    # getting the batch sizes for each iter
    q, r = divmod(len(dataset), config.test_batch_size)
    all_batches = [config.test_batch_size for i in range(q)]
    all_batches.append(r)
    n_batches = len(all_batches)
    
    with progressbar.ProgressBar(max_value=len(dataset)) as bar:
        for i in range(n_batches):
            # Populate batch
            bs = all_batches[i]
            dims = dataset[0][0].unsqueeze(0).expand(bs, -1, -1, -1).shape
            X = torch.zeros(dims)
            for j in range(bs):
                idx = sum(all_batches[:i]) + j
                X[j] = dataset[idx][0]
                
            
            # Predict batch    
            with torch.no_grad():
                X = X.to(config.device)
                a2 = model(X)
                if isinstance(a2, tuple):
                    a2 = a2[0] 
                y_pred = F.log_softmax(a2, dim=1) # log prob
                y_preds.append(y_pred.max(1)[1].cpu().detach().numpy())
                y_probs.append(torch.exp(y_pred).cpu().detach().numpy()) # prob
                bar.update(idx)
            
    y_preds = np.concatenate(y_preds, 0)
    y_probs = np.concatenate(y_probs, 0)
    assert(len(img_paths) == len(y_preds) == len(y_probs))
    
    return img_paths, y_preds, y_probs