from argparse import Namespace
import os
import torchmetrics
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.file_utils import save_pkl
from utils.utils import get_optim, get_split_loader
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

    if args.model_type =='snn':
        from models.model_genomic import SNN
        model_dict = {'n_classes': args.n_classes , 'input_dim' : args.topk}
        model = SNN(**model_dict)
    elif args.model_type == 'dtft':
        from models.dtft import DTFT
        model = DTFT(n_classes=args.n_classes)
    elif args.model_type == 'TransMIL':
        from models.TransMIL import TransMIL
        model = TransMIL(n_classes=args.n_classes)
    elif args.model_type == 'abmil':
        from models.abmil import ABMIL
        model = ABMIL()
    elif args.model_type == 'WIKG':
        from models.WIKG import WiKG
        model = WiKG(dim_in=1024, dim_hidden=256, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn')
    elif args.model_type == 'rrt':
        from models.RRT import RRTMIL
        model = RRTMIL(n_classes=args.n_classes)
    elif args.model_type == 'DAMLN':
        from models.DAMLN import DAMLN
        model = DAMLN(n_classes=args.n_classes)
    elif args.model_type == 'Porpoise':
        from models.Porpoise import PorpoiseMMF
        model = PorpoiseMMF(omic_input_dim=args.topk, path_input_dim=1024, n_classes=args.n_classes)
    elif args.model_type == 'clam_mb':
        from models.CLAM_MB import CLAM_MB
        model = CLAM_MB(n_classes=args.n_classes)
    elif args.model_type == 'clam_sb':
        from models.CLAM_MB import CLAM_SB
        model = CLAM_SB(n_classes=args.n_classes)
    elif args.model_type == 'mcat':
        from models.model_coattn import MCAT_Surv
        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out, 'fusion': args.fusion}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'cmta':
        from models.cmta import CMTA
        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out, 'fusion': args.fusion}
        model = CMTA(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.cuda()
    
    if model_weights:
        model.load_state_dict(torch.load(os.path.join(args.results_dir , model_weights)))
        # model = torch.load(os.path.join(args.results_dir , model_weights))



    test_loader = get_split_loader(test_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    test_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.CohenKappa(task='binary', num_classes = args.n_classes)])
    test_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average='macro')

    if args.mode == 'multi':
        if args.model_type == 'mcat':
            from trainer.coattn_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
            from torch.nn import MSELoss
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
        elif args.model_type == 'cmta':
            from trainer.cmta_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
        elif args.model_type == 'Porpoise':
            from trainer.porpoise_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
        else:
            raise NotImplementedError

    elif args.mode == 'omic':
        if args.model_type == 'snn':
            from trainer.snn_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
    elif args.mode == 'path':
        if args.model_type in ['abmil', 'dtft', 'cmta_path', 'TransMIL', 'WIKG', 'rrt', 'DAMLN']:
            from trainer.mil_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, test_AP, metrics, writer, loss_fn, args)
        elif args.model_type in ['clam_mb', 'clam_sb']:
            from trainer.clam_trainer import test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, test_AP, metrics, writer, loss_fn, args)
        elif args.model_type == 'DAMLN':
            from trainer.coattn_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
            test_loss, test_auc, test_ap, metrics, test_stop = test_classification_coattn(model, test_loader, test_AUROC, metrics, test_AP, writer, loss_fn, args)
    else:
        raise NotImplementedError
    test_dict = {
                'test_loss':test_loss, 
                'test_auc':test_auc.item(), 
                'test_ap': test_ap.item(),
                'test_acc': metrics['BinaryAccuracy'].item(), 
                'BinaryPrecision':metrics['BinaryPrecision'].item(),
                'BinaryRecall':metrics['BinaryRecall'].item(), 
                'BinarySpecificity':metrics['BinarySpecificity'].item(), 
                'BinaryF1Score':metrics['BinaryF1Score'].item(),
                'BinaryCohenKappa' :metrics['BinaryCohenKappa'].item()
                }

    if args.log_data:
        writer.close()
    test_print_results = {'result_auc': test_auc.cpu(), 'result_ap': test_ap.cpu(), 'result_acc': metrics['BinaryAccuracy'].cpu(), 'result_precision': metrics['BinaryPrecision'].cpu(), 'result_recall': metrics['BinaryRecall'].cpu(), 'result_specificity': metrics['BinarySpecificity'].cpu(), 'result_f1': metrics['BinaryF1Score'].cpu()}
    print("test auc: {:.4f}".format(test_auc), 'test_ap: {:.4f}'.format(test_ap))
    # print(test_print_results)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}\n'.format(test_auc))

    return test_dict, test_print_results




def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    

    else:
        writer = None

    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif args.bag_loss == 'L1Loss':
        loss_fn = nn.L1Loss()
    elif args.bag_loss == 'MSELoss':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError



    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    print('Done!')
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type =='snn':
        from models.model_genomic import SNN
        model_dict = {'n_classes': args.n_classes , 'input_dim' : args.topk}
        model = SNN(**model_dict)
    elif args.model_type == 'dtft':
        from models.dtft import DTFT
        model = DTFT(n_classes=args.n_classes)
    elif args.model_type == 'TransMIL':
        from models.TransMIL import TransMIL
        model = TransMIL(n_classes=args.n_classes)
    elif args.model_type == 'abmil':
        from models.abmil import ABMIL
        model = ABMIL(n_classes=args.n_classes)
    elif args.model_type == 'rrt':
        from models.RRT import RRTMIL
        model = RRTMIL(n_classes=args.n_classes)
    elif args.model_type == 'WIKG':
        from models.WIKG import WiKG
        model = WiKG(dim_in=1024, dim_hidden=256, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn')
    elif args.model_type == 'DAMLN':
        from models.DAMLN import DAMLN
        model = DAMLN(n_classes=args.n_classes)

    elif args.model_type == 'Porpoise':
        from models.Porpoise import PorpoiseMMF
        model = PorpoiseMMF(omic_input_dim=args.topk, path_input_dim=1024, n_classes=args.n_classes)
    elif args.model_type == 'clam_mb':
        from models.CLAM_MB import CLAM_MB
        model = CLAM_MB(n_classes=args.n_classes)
    elif args.model_type == 'clam_sb':
        from models.CLAM_MB import CLAM_SB
        model = CLAM_SB(n_classes=args.n_classes)
    elif args.model_type == 'mcat':
        from models.model_coattn import MCAT_Surv
        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out,'fusion': args.fusion}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'cmta':
        from models.cmta import CMTA
        model_dict = {'n_classes': args.n_classes , 'topk' : args.topk, 'dropout': args.drop_out, 'fusion': args.fusion}
        model = CMTA(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.cuda()
    
    if args.load_model:
        model.load_state_dict(torch.load(args.path_load_model))
    print('Done!')
    
    print('\nInit optimizer ...', end=' ')
    try :
        optimizer
    except NameError:
        optimizer = get_optim(model, args)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.train_mode == 'auc':
            early_stopping = EarlyStopping(warmup=0, patience=5, stop_epoch=5, verbose = True, metric='auc')
        elif args.train_mode == 'loss':
            early_stopping = EarlyStopping(warmup=0, patience=5, stop_epoch=5, verbose = True, metric='loss')
        else:
            raise NotImplementedError
    else:
        early_stopping = None
    
    max_auc = 0.
    epoch_max_auc = 0
    best_val_dict = {}
    AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    val_AUROC = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = args.n_classes, average = 'micro'),
                                                torchmetrics.Precision(task='binary', num_classes = args.n_classes),
                                                torchmetrics.Recall(task='binary', num_classes = args.n_classes,
                                                                average = 'macro'),
                                                torchmetrics.Specificity(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes),
                                                torchmetrics.F1Score(task='binary', average = 'macro',
                                                                    num_classes = args.n_classes)])
    train_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average='macro')
    val_AP = torchmetrics.AveragePrecision(task='binary', num_classes=args.n_classes, average='macro')
    val_metrics = metrics.clone()
    print("running with {} {}".format(args.model_type, args.mode))
    iter_num = 1
    for epoch in range(args.max_epochs):
        if args.mode == 'multi':
            if args.model_type == 'mcat':
                from trainer.coattn_trainer import train_loop_classification_coattn, validate_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args)
                if val_stop:
                    break
            elif args.model_type == 'cmta':
                from trainer.cmta_trainer import train_loop_classification_coattn, validate_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args)
                if val_stop:
                    break
            elif args.model_type == 'Porpoise':
                from trainer.porpoise_trainer import train_loop_classification_coattn, validate_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args)
                if val_stop:
                    break
            else:
                raise NotImplementedError

        elif args.mode == 'omic':
            if args.model_type == 'snn':
                from trainer.snn_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args.gc, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args.results_dir, args)
                if val_stop:
                    break
        elif args.mode == 'path':
            if args.model_type in ['abmil', 'dtft', 'cmta_path', 'TransMIL', 'WIKG', 'rrt']:
                from trainer.mil_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args.gc, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args.results_dir, args)
                if val_stop:
                    break
            elif args.model_type in ['clam_mb', 'clam_sb']:
                from trainer.clam_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args.gc, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args.results_dir, args)
                if val_stop:
                    break
            elif args.model_type == 'DAMLN':
                from trainer.DAMLN_trainer import train_loop_classification_coattn, validate_classification_coattn, test_classification_coattn
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, scheduler, AUROC, train_AP, metrics, writer, loss_fn, args.gc, args)
                val_loss, val_auc, val_ap, all_val_metrics, val_stop = validate_classification_coattn(cur, epoch, model, val_loader, val_AUROC, val_AP, val_metrics, early_stopping, writer, loss_fn, args.results_dir, args)
                if val_stop:
                    break
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


        if val_auc > max_auc:
            max_auc = val_auc
            best_val_dict = {
                            'val_loss':val_loss, 
                            'val_auc':val_auc.item(), 
                            'val_ap' : val_ap.item(),
                            'val_acc': all_val_metrics['BinaryAccuracy'].item(), 
                            'BinaryPrecision':all_val_metrics['BinaryPrecision'].item(),
                            'BinaryRecall':all_val_metrics['BinaryRecall'].item(),
                            'BinarySpecificity':all_val_metrics['BinarySpecificity'].item(),
                            'BinaryF1Score':all_val_metrics['BinaryF1Score'].item()
                            }
            
        val_AUROC.reset()
        AUROC.reset()
        val_metrics.reset()
        metrics.reset()
        train_AP.reset()
        val_AP.reset()
                # torch.cuda.empty_cache()
        

    if args.log_data:
        writer.close()
    val_print_results = {'result_auc': max_auc.cpu(), 'result_ap': val_ap.cpu()}
    print("================= summary of fold {} ====================".format(cur))
    print("val auc: {:.4f}, val ap: {:.4f}".format(max_auc, val_ap))

    return best_val_dict, val_print_results