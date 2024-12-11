from __future__ import print_function
import argparse
import os
import random
import sys
import torch
from timeit import default_timer as timer
import numpy as np

# Internal Imports
from dataset.path_omic_dataset_eval_tcga import classification_dataset
from utils.file_utils import save_pkl
from utils.core_utils_eval import test
from utils.utils_eval import get_custom_exp_code
import pandas as pd

torch.autograd.set_detect_anomaly(True)
# export CUDA_VISIBLE_DEVICES=0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# PyTorch Imports
import torch
#cp -r /home/qiyuan/nvme0n1/ZQB/CLAM-master/TCGA-BRCA/x40_512/Virchow_features/pt_files /home/qiyuan/nvme0n1/TCGA-BRCA/Virshow_fearures/pt_files
#   
# /home/qiyuan/nvme0n1/TCGA-BRCA/uni_features/pt_files
# /home/qiyuan/nvme0n1/TCGA-BRCA/Virchow_fearures
# /home/qiyuan/nvme0n1/ZQB/PAMA-main/data/feat
### Training settings
parser = argparse.ArgumentParser(
    description='Configurations for classification Analysis on TCGA-BRCA Data.')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default=r'/home/qiyuan/nvme0n1/TCGA-BRCA/uni_features/pt_files',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1,
                    help='Random seed for reproducible experiment (default: 1)') 
parser.add_argument('--n_classes', 			 type=int, default=2,
                    help='classes (default: 2)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--results_dir',     type=str, default=r'./eval_brca_result',
                    help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default=r'5foldcv_13',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='ER',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
parser.add_argument('--log_data',        action='store_true', 
                    help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--load_model',        action='store_true',
                    default=False, help='whether to load model')
parser.add_argument('--path_load_model', type=str,
                    default='brca_result/5foldcv_loss/TransMIL_CrossEntropyLoss_5foldcv_gc32_bilinear/ER_TransMIL_CrossEntropyLoss_5foldcv_gc32_bilinear_s1', help= 'path of ckpt for loading')
parser.add_argument('--stage', default=r'test', type=str, choices=['train', 'test'], help='train or test')
parser.add_argument('--topk', default = 8192, type=int, help='topk omic features to use') # 17920 8192 4096 2048 1024 512
parser.add_argument('--pvalues_path', default=r'cox_model_summary.csv', type=str, help='pvalues_path')
parser.add_argument('--omic_path', default=r'data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.csv', type=str, help='omic_path')


### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['mul_kd', 'mul_abmil_snn', 'abmil_contrast', 'batch_kd', 'abmil', 'dtft', 'clam_sb', 'clam_mb', 'abmil_snn',
                                                            'TransMIL', 'fof', 'cmta_path', 'mamba2mil', 'mamba2mil_snn'], 
                    default=r'TransMIL', help='Type of model (Default: motcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'multi'],
                    default=r'path', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=[
                    'moe', 'concat', 'bilinear'], default=r'bilinear', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=True,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--drop_out',        action='store_true',
                    default=0.25, help='Enable dropout (p=0.25)')
parser.add_argument('--temp',   action='store_true', default=8, help='Temperature coefficient (t=8)')


### Optimizer Parameters + classification Loss Function
parser.add_argument('--opt',             type=str,
                    choices=['adam', 'sgd', 'adamw'], default='adamw')
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int,
                    default=16, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=100,
                    help='Maximum number of epochs to train (default: 100)')
parser.add_argument('--lr',				 type=float, default=2e-4,
                    help='Learning rate (default: 0.0002)')
parser.add_argument('--bag_loss',        type=str, choices=['L1Loss', 'CrossEntropyLoss', 'MSELoss']
                    , default=r'CrossEntropyLoss', help='slide-level classification loss function (default: nll_surv)')
parser.add_argument('--reg', 			 type=float, default=1e-5,
                    help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                    help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true',
                    default=True, help='Enable early stopping')
parser.add_argument('--start', type=int, default=0, help='start fold')

args = parser.parse_args()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_Classification'

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'data_root_dir': args.data_root_dir,
            'num_splits': args.k,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,}

print('\nLoad Dataset')


# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)


exp_code = str(args.exp_code) + '_s{}'.format(args.seed)

print("===="*30)
print("Experiment Name:", exp_code)
print("===="*30)

args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)
print("logs saved at ", args.results_dir)
# 将参数保存到文件
with open(os.path.join(args.results_dir, 'configure.txt'), 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')
if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

# Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

def main(args):
    # Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)


    folds = args.k
    start = args.start
    
    # Start 5-Fold CV Evaluation.
    # run_folds = folds
    val_summary_all_folds = {}
    test_summary_all_folds = {}
    test_list = []
    val_list = []
    for i in range(start, folds):
        start_t = timer()
        seed_torch(args.seed)
        args.results_pt_path = os.path.join(
            args.results_dir, 's_{}_max_auc_checkpoint.pt'.format(i))

        if args.stage == 'test':
            test_dataset = classification_dataset(wsi_dir=args.data_root_dir, 
                                        topk = args.topk, 
                                        pvalues_path = args.pvalues_path, 
                                        omic_path = args.omic_path,
                                        state ='test',
                                        data_splits_csv = os.path.join(args.split_dir, 'splits_{}.csv'.format(i)),
                                        transform=None,
                                        mode='path',
                                        num_classes=args.n_classes)
            args.results_pkl_path = os.path.join(args.results_dir, 'result_test.pkl')
            print('testing: {},'.format(len(test_dataset)))
            datasets = test_dataset
        else:
            raise NotImplementedError 

        
        # Run Train-Val on classification Task.
        if args.stage == 'test':
            test_summary_results, test_print_results = test(datasets, i, "s_{}_max_auc_checkpoint.pt".format(i), args)
            test_list.append(test_summary_results)
            test_summary_all_folds[i] = test_print_results
        else:
            raise NotImplementedError 


    #train
    # test
    if args.stage == 'test':
        end_t = timer()
        save_pkl(args.results_pkl_path, test_list)
        print('Fold %d Time: %f seconds' % (i, end_t - start_t))
        print('=============================== summary ===============================')

        result_auc = []
        for i, k in enumerate(test_summary_all_folds):
            auc = test_summary_all_folds[k]['result']
            print("Fold {}, auc: {:.4f}".format(k, auc))
            result_auc.append(auc)
        result_auc = np.array(result_auc)
        print("test Avg auc of {} folds: {:.4f}, stdp: {:.4f}, stds: {:.4f}".format(
            len(test_summary_all_folds), result_auc.mean(), result_auc.std(), result_auc.std(ddof=1)))
        
        df = pd.DataFrame(test_list)
        mean_values = df.mean()
        df.loc['mean'] = mean_values
        df.to_csv(os.path.join(args.results_dir, 'result.csv'), index=False)
        
    else:
        raise NotImplementedError 


if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))