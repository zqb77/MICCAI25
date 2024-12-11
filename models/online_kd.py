#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_coattn.py
@Time    :   2022/07/07 16:43:59
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils.model_utils import *



###########################
### MCAT Implementation ###
###########################
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        # x = x.unsqueeze(0)
        ## x: N x L
        # x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights( A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N



class ABMIL_SNN(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, topk=1024,
                dropout=0.25):
        super(ABMIL_SNN, self).__init__()
        self.fusion = fusion
        self.topk = topk
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.dropout = dropout

        feature_dim = 256


        self.wsi_net = nn.Sequential(nn.Linear(1024, feature_dim), 
                                    nn.ReLU(), 
                                    nn.Dropout(0.25))

        self.snn = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        self.attention_path = Attention_Gated(L=256, D=128, K=1)
        self.attention_snn = Attention_Gated(L=256, D=128, K=1)

        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(feature_dim*2, 512), nn.ReLU(), nn.Linear(512, feature_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=feature_dim, dim2=feature_dim, scale_dim1=8, scale_dim2=8, mmhid=feature_dim)
        else:
            self.gating_network = nn.Sequential(
                    nn.Linear(1024, 2),
                    nn.Softmax(dim=1)
                )
        
        ### Classifier
        self.classifier = nn.Linear(feature_dim, n_classes)


    def forward(self, **kwargs):
        x_path = kwargs['x_path'] # (num_patch, 1024)
        x_omic = kwargs['x_omic'].reshape(-1, 512) # (1, num_omic)

        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer [1, n, 256]
        h_omic_bag = self.snn(x_omic)### omic embeddings are fed through a FC layer [1, n, 256]

        AA_path = self.attention_path(h_path_bag)
        h_path_bag = torch.mm(AA_path, h_path_bag)
        AA_snn = self.attention_snn(h_omic_bag)
        h_omic_bag = torch.mm(AA_snn, h_omic_bag)
        

        if self.fusion == "concat":
            fusion = self.mm(torch.concat((h_path_bag,h_omic_bag),dim=1)) # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(h_path_bag,h_omic_bag)  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
        
        logits = self.classifier(fusion.reshape(1, -1)) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        return  logits, Y_prob, Y_hat, fusion

class ABMIL(nn.Module):
    def __init__(self, L=256, D=128, K=1, n_classes=2, dropout=0):
        super(ABMIL, self).__init__()
        
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, n_classes, dropout)
        self._fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        init_max_weights(self)
    def forward(self, **kwargs): ## x: N x L
        h  = kwargs['x_path'] #[B, n, 1024]
        h = self._fc1(h) #[B, n, 256]
        x = h.squeeze(0)
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        logits = self.classifier(afeat) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits, Y_prob, Y_hat, afeat

class ABMIL_omic(nn.Module):
    def __init__(self, L=256, D=128, K=1, n_classes=2, dropout=0):
        super(ABMIL_omic, self).__init__()
        
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, n_classes, dropout)
        self._fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        init_max_weights(self)
    def forward(self, **kwargs): ## x: N x L
        h  = kwargs['x_path'] #[B, n, 1024]
        # h = self._fc1(h) #[B, n, 256]
        x = h.squeeze(0)
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        logits = self.classifier(afeat) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits, Y_prob, Y_hat, afeat


class SNN(nn.Module):
    def __init__(self, n_classes: int=2, dropout: float=0.25, topk=1024) -> None:
        super(SNN, self).__init__()
        feature_dim = 256
        self.n_classes =  n_classes
        self.dropout = dropout
        self.topk = topk
        self.fc1 = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        
        self.abmil = ABMIL_omic(L=feature_dim, D=128, K=1, n_classes=n_classes, dropout=dropout)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.topk, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout, inplace=False)
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ELU(),
        #     nn.AlphaDropout(p=dropout, inplace=False)
        # )

        self.classifier = nn.Linear(feature_dim, n_classes)

        init_max_weights(self)

    def forward(self, feat):
        feat = feat.reshape(-1, 512).unsqueeze(0)
        feat = self.fc1(feat)
        logits, Y_prob, Y_hat, afeat = self.abmil(x_path=feat)

        # feat = self.fc2(feat)
        # logits = self.classifier(feat)
        # Y_prob = F.softmax(logits, dim=1)
        # Y_hat = torch.argmax(logits, dim=1)
        return logits, Y_prob, Y_hat, afeat
