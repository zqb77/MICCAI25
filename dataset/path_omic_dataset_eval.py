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



    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        
        sample = self.names[idx]
        label = self.labels[idx]
        # slide_id = self.get_slide_ids(sample)[0]   
        slide_id = sample
        wsi_path = os.path.join(self.wsi_dir, slide_id + '.pt')
        wsi_feat =  torch.load(wsi_path)
        wsi_feat = wsi_feat.clone().detach().float()
        feat = [wsi_feat]
        label = int(label)
        return feat, label
    
    def getlabel(self, idx):
        return self.labels[idx]


    def get_slide_ids(self, case_id):
        return self.case_to_slide.get(case_id, [])

if __name__ == '__main__':
    dataset = classification_dataset()
    for feat, label in dataset:
        print(feat, label)
        break




