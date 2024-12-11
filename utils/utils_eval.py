import torch
import numpy as np
import torch.nn as nn
import pdb
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from torch.utils.data.dataloader import default_collate

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)




def collate_MIL_classification(batch):
    img = torch.cat([item[0][0] for item in batch], dim = 0)

    label = torch.LongTensor([item[1] for item in batch])

    return [img, label]

# def collate_omic_classification(batch):
#     img = torch.cat([item[0][0] for item in batch], dim = 0)

#     label = torch.LongTensor([item[1] for item in batch])

#     return [img, label]


# def collate_multi_classification(batch):
#     img = torch.cat([item[0][0] for item in batch], dim = 0)
#     omic = torch.cat([item[0][1] for item in batch], dim = 0)

#     label = torch.LongTensor([item[1] for item in batch])

#     return [img, omic, label]

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    # if mode == 'multi':
    #     collate = collate_multi_classification
    # elif mode == 'omic':
    #     collate = collate_omic_classification
    # elif mode == 'path':
    #     collate = collate_MIL_classification
    # else:
    #     raise NotImplementedError
    collate = collate_MIL_classification
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}

    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate,**kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[int(y)]                                  

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)




def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg


def get_custom_exp_code(args):
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    ### Model Type
    if args.model_type == 'snn':
        param_code += 'SNN'
    elif args.model_type == 'abmil':
        param_code += 'ABMIL'
    elif args.model_type == 'clam_mb':
        param_code += 'CLAM_MB'
    elif args.model_type == 'dtft':
        param_code += 'DTFT'
    elif args.model_type == 'mcat':
        param_code += 'MCAT'
    elif args.model_type == 'reg':
        param_code += 'reg'
    elif args.model_type == 'abmil_snn':
        param_code += 'ABMIL_SNN'
    elif args.model_type == 'kd':
        param_code += 'KD'
    elif args.model_type == 'mmim':
        param_code += 'MMIM'
    elif args.model_type == 'moe':
        param_code += 'MOE'
    elif args.model_type == 'cmta':
        param_code += 'CMTA'
    elif args.model_type == 'cmta_kd':
        param_code += 'CMTA_KD'
    elif args.model_type == 'fof':
        param_code += 'FOF'
    elif args.model_type == 'abmil_kd':
        param_code += 'ABMIL_KD'
    elif args.model_type == 'abmil_snn_kd':
        param_code += 'ABMIL_SNN_KD'
    elif args.model_type == 'abmil_snn_abmil_kd':
        param_code += 'ABMIL_SNN_ABMIL_KD'
    elif args.model_type == 'abmil_mmim':
        param_code += 'ABMIL_MMIM'
    elif args.model_type == 'TransMIL':
        param_code += 'TransMIL'
    elif args.model_type == 'clam_sb':
        param_code += 'CLAM_SB'
    elif args.model_type == 'batch_kd':
        param_code += 'BATCH_KD'
    elif args.model_type == 'online_kd':
        param_code += 'ONLINE_KD'
    elif args.model_type == 'rrt':
        param_code += 'RRT'
    elif args.model_type == 'WIKG':
        param_code += 'WIKG'
    else:
        raise NotImplementedError
    

    ### Loss Function
    param_code += '_%s' % args.bag_loss
    # param_code += '_a%s' % str(args.alpha_surv)

    ### Learning Rate
    if args.lr != 2e-4:
        param_code += '_lr%s' % format(args.lr, '.0e')

    ### L1-Regularization
    if args.reg_type != 'None':
        param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    ### Batch Size
    if args.batch_size != 1:
      param_code += '_b%s' % str(args.batch_size)

    ### Gradient Accumulation
    if args.gc != 1:
      param_code += '_gc%s' % str(args.gc)

    if args.mode == 'omic' or args.mode == 'multi':
        param_code += '_topk%s' % args.topk

    ### Fusion Operation
    if args.fusion != "None":
        param_code += '_' + args.fusion


    args.exp_code = exp_code + "_" + param_code
    if args.model_type == 'online_kd' or args.model_type == 'abmil_snn_abmil_kd':
        args.exp_code += '_temp' + str(args.temp)
    args.param_code = param_code
    args.dataset_path = dataset_path


    return args