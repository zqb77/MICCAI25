from argparse import Namespace
import os
import torchmetrics
import numpy as np
import torch
from utils.utils_eval import get_split_loader
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False, metric='auc'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.auc_max = np.Inf
        self.metric = metric

    def __call__(self, epoch, auc, model, ckpt_name = 'checkpoint.pt'):

        if self.metric == 'auc':
            score = auc

            if epoch < self.warmup:
                pass
            elif self.best_score is None:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
            elif score <= self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
                self.counter = 0
        elif self.metric == 'loss':
            score = -auc

            if epoch < self.warmup:
                pass
            elif self.best_score is None:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
            elif score <= self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(auc, model, ckpt_name)
                self.counter = 0

    def save_checkpoint(self, auc, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.metric == 'auc':
                print(f'auc increased ({self.auc_max:.6f} --> {auc:.6f}).  Saving model ...')
            elif self.metric == 'loss':
                print(f'val_loss decreased ({self.auc_max:.6f} --> {auc:.6f}).  Saving model ...')

        torch.save(model.state_dict(), ckpt_name)
        self.auc_max = auc



def test(datasets: tuple, cur: int, model_weights, args: Namespace):
    """   
        train for a single fold
    """
    # print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    else:
        writer = None


    test_split = datasets

    if args.bag_loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif args.bag_loss == 'L1Loss':
        loss_fn = nn.L1Loss()
    elif args.bag_loss == 'MSELoss':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError

    

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'topk': args.topk}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'abmil' or args.model_type == 'online_kd':
        from models.abmil import ABMIL
        model_dict = {'n_classes': args.n_classes , 'droprate': args.drop_out}
        model = ABMIL(**model_dict)
    elif args.model_type == 'WIKG':
        from models.WIKG import WiKG
        model = WiKG(dim_in=1024, dim_hidden=256, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn')
    elif args.model_type == 'rrt':
        from models.RRT import RRTMIL
        model = RRTMIL(n_classes=args.n_classes)
    elif args.model_type == 'dtft':
        from models.dtft import DTFT
        model_dict = {'n_classes': args.n_classes , 'droprate': args.drop_out}
        model = DTFT(**model_dict)
    elif args.model_type == 'clam_mb':
        from models.CLAM_MB import CLAM_MB
        model = CLAM_MB(n_classes=args.n_classes)
    elif args.model_type == 'clam_sb':
        from models.CLAM_MB import CLAM_SB
        model = CLAM_SB(n_classes=args.n_classes)
    elif args.model_type == 'TransMIL':
        from models.TransMIL import TransMIL
        model = TransMIL(n_classes=args.n_classes)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.cuda()
        

    if not args.path_load_model:
        model.load_state_dict(torch.load(os.path.join(args.results_dir , model_weights)))
    




    test_loader = get_split_loader(test_split, testing = False, mode=args.mode, batch_size=args.batch_size)
    test_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes)],
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes))
    test_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average='macro')


    if args.mode == 'omic':
        if args.model_type == 'snn':
            from trainer.snn_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
            test_loss, test_auc, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
    elif args.mode == 'path' or args.mode == 'multi':
        if args.model_type in ['abmil', 'dtft', 'cmta_path', 'TransMIL', 'online_kd', 'rrt', 'WIKG']:
            from trainer.mil_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, test_AP, metrics, writer, loss_fn, args)
        if args.model_type in ['clam_mb', 'clam_sb']:
            from trainer.clam_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, test_AP, metrics, writer, loss_fn, args)
    else:
        raise NotImplementedError
    test_dict = {
                'test_loss':test_loss, 
                'test_auc':test_auc.item(), 
                'test_ap':test_ap.item(),
                'test_acc': metrics['BinaryAccuracy'].item(), 
                'BinaryPrecision':metrics['BinaryPrecision'].item(),
                'BinaryRecall':metrics['BinaryRecall'].item(), 
                'BinarySpecificity':metrics['BinarySpecificity'].item(), 
                'BinaryF1Score':metrics['BinaryF1Score'].item(),
                'BinaryCohenKappa' : metrics['BinaryCohenKappa'].item(),
                }

    if args.log_data:
        writer.close()
    test_print_results = {'result': test_auc.cpu()}
    print("test auc: {:.4f}".format(test_auc))
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}\n'.format(test_auc))

    return test_dict, test_print_results

