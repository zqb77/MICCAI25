import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
import pandas as pd




class classification_dataset(Dataset):
    def __init__(self, wsi_dir=None, topk = 512, pvalues_path = r'cox_model_summary.csv', omic_path = r'brca_gene_clean_log_normalized.csv', 
                    state ='train', data_splits_csv = r'splits/5foldcv/ER/splits_0.csv', transform=None, mode=r'omic', num_classes=2):
        self.transform = transform
        self.mode = mode
        self.omic_path = omic_path
        self.wsi_dir = wsi_dir
        self.topk = topk
        self.pvalues_path = pvalues_path
        self.data_splits_csv = data_splits_csv
        self.state = state
        self.num_classes = num_classes

        
        #data splits
        self.data_splits_csv = pd.read_csv(self.data_splits_csv, index_col=0, dtype=str)
        if state == 'train':
            self.names = self.data_splits_csv.loc[:, 'train'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.names = self.data_splits_csv.loc[:, 'val'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.names = self.data_splits_csv.loc[:, 'test'].dropna()
            self.labels = self.data_splits_csv.loc[:, 'test_label'].dropna()

        # omic
        if self.mode == 'omic' or self.mode == 'multi':
            pvalues = pd.read_csv(self.pvalues_path, index_col=0)

            top_k_covariates = pvalues.nsmallest(self.topk, 'p')['covariate'].values

            omic = pd.read_csv(self.omic_path, index_col=0)
            omic.reset_index(inplace=True)

            omic = omic.drop_duplicates(subset='case_id', keep='first')
            omic.set_index('case_id', inplace=True)

            self.omic = omic[top_k_covariates]
            self.omic_sizes = self.omic.shape[1]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(np.array(self.labels) == str(i))[0]

        self.case_to_slide = self._create_case_to_slide_mapping()


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):


        if self.mode == 'omic':
            sample = self.names[idx]
            label = self.labels[idx]
            omic_feat = torch.tensor(self.omic.loc[sample].values, dtype=torch.float32)
            feat = [omic_feat]
        elif self.mode == 'path':
            sample = self.names[idx]
            label = self.labels[idx]
            slide_id = self.get_slide_ids(sample)[0]        # Assuming there is only one slide per case
            wsi_path = os.path.join(self.wsi_dir, slide_id)
            wsi_feat =  torch.load(wsi_path)
            wsi_feat = wsi_feat.clone().detach().float()
            feat = [wsi_feat]
        elif self.mode == 'multi':
            sample = self.names[idx]
            # print(idx , sample)
            label = self.labels[idx]
            slide_id = self.get_slide_ids(sample)[0]        # Assuming there is only one slide per case
            wsi_path = os.path.join(self.wsi_dir, slide_id)
            wsi_feat =  torch.load(wsi_path)
            omic_feat = torch.tensor(self.omic.loc[sample].values, dtype=torch.float32)
            feat = [wsi_feat, omic_feat]
        else:
            raise ValueError('mode must be either omic, wsi or multi')
        label = int(label)
        return feat, label, idx
    
    def getlabel(self, idx):
        return self.labels[idx]

    def _create_case_to_slide_mapping(self):
        case_to_slide = {}
        pt_files = [f for f in os.listdir(self.wsi_dir) if f.endswith('.pt')]
        for pt_file in pt_files:
            case_id = pt_file[:12]  # Assuming the first 12 characters are the case_id
            slide_id = pt_file  # The entire filename is the slide_id
            if case_id in case_to_slide:
                case_to_slide[case_id].append(slide_id)
            else:
                case_to_slide[case_id] = [slide_id]
        return case_to_slide

    def get_slide_ids(self, case_id):
        return self.case_to_slide.get(case_id, [])

if __name__ == '__main__':
    dataset = classification_dataset()
    for feat, label in dataset:
        print(feat, label)
        break




