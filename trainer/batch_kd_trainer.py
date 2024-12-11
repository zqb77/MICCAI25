import numpy as np
import torch
import os
import torch.nn.functional as F
from utils.loss.CRD_criterion_v10 import assign_sample_weights

all_afeat = []
all_feat_abmil_snn = []
all_feat_abmil = []
all_feat_snn = []

all_logits = []
all_logits_labels_abmil_snn = []
all_logits_labels_abmil = []
all_logits_labels_snn = []

alphas = []
betas = []
gammas = []
indexs = []
sample_idxs = []
labels = []
student_preds = []
teacher1_preds = []
teacher2_preds = []
teacher3_preds = []
KRC_cors1 = []
KRC_cors2 = []
KRC_cors3 = []
def train_loop_classification_coattn(epoch, model, tea_model_abmil_snn, tea_model_abmil, tea_model_snn, loader, optimizer, scheduler, train_class_idx, 
                                    inter_fn, logtis_fn, AUROC, AP, metrics, iter_num, writer=None, loss_fn=None, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)
    if not callable(inter_fn):
        if isinstance(inter_fn, list):
            for inter_fns in inter_fn:
                inter_fns = inter_fns.to(device)
        elif isinstance(inter_fn, bool):
            pass
        else:
            inter_fn = inter_fn.to(device)

    if not callable(logtis_fn):
        if isinstance(logtis_fn, list):
            for logtis_fns in logtis_fn:
                logtis_fns = logtis_fns.to(device)
        elif isinstance(logtis_fn, bool):
            pass
        else:
            logtis_fn = logtis_fn.to(device)

    tea_model_abmil_snn = tea_model_abmil_snn.to(device) if args.tea_model1 else None
    tea_model_abmil = tea_model_abmil.to(device) if args.tea_model2 else None
    tea_model_snn = tea_model_snn.to(device) if args.tea_model3 else None
    print('\n')
    k = args.nce_k
    global all_afeat, all_feat_abmil_snn, all_feat_abmil, all_feat_snn
    global alphas, betas, gammas, student_preds, teacher1_preds, teacher2_preds, teacher3_preds, indexs, sample_idxs, labels
    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):

        assert index in train_class_idx[label[0]]
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.type(torch.FloatTensor).to(device)

        all_neg_idx = list(range(0, len(loader)))
        all_neg_idx.remove(index[0])
        replace = True if k > len(all_neg_idx) else False
        neg_idx = np.random.choice(all_neg_idx, k, replace=replace) # 选择k个负样本
        sample_idx = np.hstack([index, neg_idx])

        w = 0.
        label = label.type(torch.LongTensor).to(device)
        logits, Y_prob, Y_hat, afeat= model(x_path=data_WSI)
        student_preds.append(F.softmax(logits, dim=0))
        with torch.no_grad():
            if args.tea_model1:
                logits_labels_abmil_snn, _, _, afeat_abmil_snn= tea_model_abmil_snn(x_path=data_WSI, x_omic=data_omic)
                alpha = F.softmax(torch.cat([logits[0], logits_labels_abmil_snn[0]], dim=0)/args.temp, dim=0)
                alpha = alpha[2 + label.item()]/alpha[label.item()]
                teacher1_preds.append(F.softmax(logits_labels_abmil_snn, dim=1))
                # aa = spearmanr(afeat.squeeze().cpu().detach().numpy(), afeat_abmil_snn.squeeze().cpu().detach().numpy())[0]
                # KRC_cors1.append(spearmanr(afeat.squeeze().cpu().detach().numpy(), afeat_abmil_snn.squeeze().cpu().detach().numpy())[0] > w)
            else:
                alpha, logits_labels_abmil_snn, afeat_abmil_snn = 0, 0, 0
            if args.tea_model2:
                logits_labels_abmil, _, _, feat_abmil = tea_model_abmil(x_path=data_WSI)
                beta = F.softmax(torch.cat([logits[0], logits_labels_abmil[0]], dim=0)/args.temp, dim=0)
                beta = beta[2 + label.item()]/beta[label.item()]
                teacher2_preds.append(F.softmax(logits_labels_abmil, dim=1))
                # bb = spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_abmil.squeeze().cpu().detach().numpy())[0]
                # KRC_cors2.append(spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_abmil.squeeze().cpu().detach().numpy())[0] > w)
            else:
                beta, logits_labels_abmil, feat_abmil = 0, 0, 0
            if args.tea_model3:
                logits_labels_snn, _, _, feat_snn = tea_model_snn(feat=data_omic) 
                gamma = F.softmax(torch.cat([logits[0], logits_labels_snn[0]], dim=0)/args.temp, dim=0)
                gamma = gamma[2 + label.item()]/gamma[label.item()]
                teacher3_preds.append(F.softmax(logits_labels_snn, dim=1))
                # cc = spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_snn.squeeze().cpu().detach().numpy())[0]
                # KRC_cors3.append(spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_snn.squeeze().cpu().detach().numpy())[0] > w)
            else:
                gamma, logits_labels_snn, feat_snn = 0, 0, 0


        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
        indexs.append(torch.tensor(index))
        sample_idxs.append(torch.tensor(sample_idx))
        labels.append(label)

        loss_class = loss_fn(logits, label)
        # kl_loss_1 = logtis_fn(logits, logits_labels_abmil_snn.detach())  if args.tea_model1 else 0
        # kl_loss_2 = logtis_fn(logits, logits_labels_abmil.detach())  if args.tea_model2 else 0
        # kl_loss3 = logtis_fn(logits, logits_labels_snn.detach())  if args.tea_model3 else 0
        # kl_loss = kl_loss_1 + kl_loss_2 + kl_loss3


        all_afeat.append(afeat)
        all_feat_abmil_snn.append(afeat_abmil_snn)
        all_feat_abmil.append(feat_abmil)
        all_feat_snn.append(feat_snn)
        all_logits.append(logits)
        all_logits_labels_abmil_snn.append(logits_labels_abmil_snn)
        all_logits_labels_abmil.append(logits_labels_abmil)
        all_logits_labels_snn.append(logits_labels_snn)
        loss = loss_class


        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss / args.gc
        train_loss += loss.item()
        loss.backward(retain_graph=True)

        if (batch_idx + 1) % args.gc == 0: 
            
            batch_alpha = torch.stack(alphas, dim=0) if args.tea_model1 else 0
            batch_beta = torch.stack(betas, dim=0) if args.tea_model2 else 0
            batch_gamma = torch.stack(gammas, dim=0) if args.tea_model3 else 0
            bsz = batch_gamma.size(0)

            all_afeat_tensor = torch.cat(all_afeat, dim=0)
            all_feat_abmil_snn_tensor = torch.cat(all_feat_abmil_snn, dim=0) if args.tea_model1 else 0
            all_feat_abmil_tensor = torch.cat(all_feat_abmil, dim=0) if args.tea_model2 else 0
            all_feat_snn_tensor = torch.cat(all_feat_snn, dim=0) if args.tea_model3 else 0

            all_logits_tensor = torch.cat(all_logits, dim=0)
            all_logits_labels_abmil_snn_tensor = torch.cat(all_logits_labels_abmil_snn, dim=0) if args.tea_model1 else 0
            all_logits_labels_abmil_tensor = torch.cat(all_logits_labels_abmil, dim=0) if args.tea_model2 else 0
            all_logits_labels_snn_tensor = torch.cat(all_logits_labels_snn, dim=0) if args.tea_model3 else 0

            if args.intermediate_loss_fn == 'SP' or args.intermediate_loss_fn == 'PKT':
            # loss_rkd_1 = inter_fn(all_afeat_tensor * batch_alpha.view(bsz, 1), all_feat_abmil_snn_tensor * batch_alpha.view(bsz, 1)) if args.tea_model1 else 0
            # loss_rkd_2 = inter_fn(all_afeat_tensor * batch_beta.view(bsz, 1), all_feat_abmil_tensor *  batch_beta.view(bsz, 1)) if args.tea_model2 else 0
            # loss_rkd_3 = inter_fn(all_afeat_tensor * batch_gamma.view(bsz, 1), all_feat_snn_tensor * batch_gamma.view(bsz, 1)) if args.tea_model3 else 0
                loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor.detach()) if args.tea_model1 else 0
                loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor.detach()) if args.tea_model2 else 0
                loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor.detach()) if args.tea_model3 else 0
                loss_rkd = (loss_rkd_1 + loss_rkd_2 + loss_rkd_3) * bsz
            elif args.intermediate_loss_fn == 'AB':
                loss_rkd1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor.detach()) if args.tea_model1 else 0
                loss_rkd2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor.detach()) if args.tea_model2 else 0
                loss_rkd3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor.detach()) if args.tea_model3 else 0
                loss_rkd = loss_rkd1 + loss_rkd2 + loss_rkd3
            elif args.intermediate_loss_fn == 'TDC':
                batch_indexs = torch.stack(indexs, dim=0).view(-1).to(device)
                batch_sample_idxs = torch.stack(sample_idxs, dim=0).to(device)
                batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
                student_preds_tensor = torch.stack(student_preds, dim=0).to(device).squeeze(1)
                teacher1_preds_tensor = torch.stack(teacher1_preds, dim=0).to(device).squeeze(1) if args.tea_model1 else 0
                teacher2_preds_tensor = torch.stack(teacher2_preds, dim=0).to(device).squeeze(1) if args.tea_model2 else 0
                teacher3_preds_tensor = torch.stack(teacher3_preds, dim=0).to(device).squeeze(1) if args.tea_model3 else 0
                teacher1_sample_weights = assign_sample_weights(student_preds_tensor, teacher1_preds_tensor, batch_labels, 1, 1) if args.tea_model1 else 0
                teacher2_sample_weights = assign_sample_weights(student_preds_tensor, teacher2_preds_tensor, batch_labels, 1, 1) if args.tea_model2 else 0
                teacher3_sample_weights = assign_sample_weights(student_preds_tensor, teacher3_preds_tensor, batch_labels, 1, 1) if args.tea_model3 else 0
                loss_rkd_1, sample_loss_rkd_1 = inter_fn[0](teacher1_sample_weights, all_afeat_tensor, all_feat_abmil_snn_tensor, batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
                loss_rkd_2, sample_loss_rkd_2 = inter_fn[1](teacher2_sample_weights, all_afeat_tensor, all_feat_abmil_tensor, batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
                loss_rkd_3, sample_loss_rkd_3 = inter_fn[2](teacher3_sample_weights, all_afeat_tensor, all_feat_snn_tensor, batch_labels, batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            elif args.intermediate_loss_fn == 'CRD':
                batch_indexs = torch.stack(indexs, dim=0).view(-1).to(device)
                batch_sample_idxs = torch.stack(sample_idxs, dim=0).to(device)
                loss_rkd_1 = inter_fn[0](all_afeat_tensor, all_feat_abmil_snn_tensor, batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
                loss_rkd_2 = inter_fn[1](all_afeat_tensor, all_feat_abmil_tensor, batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
                loss_rkd_3 = inter_fn[2](all_afeat_tensor, all_feat_snn_tensor, batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            elif args.intermediate_loss_fn == 'SP_M':
                loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor, all_logits_labels_abmil_snn_tensor) if args.tea_model1 else 0
                loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor, all_logits_labels_abmil_tensor) if args.tea_model2 else 0
                loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor, all_logits_labels_snn_tensor) if args.tea_model3 else 0
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            else:
                loss_rkd = 0
            


            if  args.logits_loss_fn == 'KL'or args.logits_loss_fn == 'BKD' or args.logits_loss_fn == 'MLKD':
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach())  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach())  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach())  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
            elif args.logits_loss_fn == 'DKD':
                batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels, args.temp)  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
            elif args.logits_loss_fn == 'NKD':
                batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
                kl_loss_1 = logtis_fn[0](all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels,)  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn[0](all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels)  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn[0](all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels)  if args.tea_model3 else 0
                
                kl_loss_1_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels,)  if args.tea_model1 else 0
                kl_loss_2_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model2 else 0
                kl_loss_3_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_1_1 + kl_loss_2_1 + kl_loss_3_1
            else:
                kl_loss = 0



            loss_kd = loss_rkd + kl_loss
            loss_kd.backward()
            optimizer.step()
            optimizer.zero_grad()
            # update_ema_variables(model, tea_model_abmil, 0.999, iter_num)
            iter_num += 1
            all_afeat.clear()
            all_feat_abmil_snn.clear()
            all_feat_abmil.clear()
            all_feat_snn.clear()
            alphas.clear()
            betas.clear()
            gammas.clear()
            indexs.clear()
            sample_idxs.clear()
            labels.clear()
            student_preds.clear()
            teacher1_preds.clear()
            teacher2_preds.clear()
            teacher3_preds.clear()
            all_logits.clear()
            all_logits_labels_abmil_snn.clear()
            all_logits_labels_abmil.clear()
            all_logits_labels_snn.clear()
            train_loss += loss_kd.item()

        # 检查是否有剩余的梯度需要更新
    if (batch_idx + 1) % args.gc != 0:
        batch_alpha = torch.stack(alphas, dim=0) if args.tea_model1 else 0
        batch_beta = torch.stack(betas, dim=0) if args.tea_model2 else 0
        batch_gamma = torch.stack(gammas, dim=0) if args.tea_model3 else 0
        bsz = batch_gamma.size(0)

        all_afeat_tensor = torch.cat(all_afeat, dim=0)
        all_feat_abmil_snn_tensor = torch.cat(all_feat_abmil_snn, dim=0) if args.tea_model1 else 0
        all_feat_abmil_tensor = torch.cat(all_feat_abmil, dim=0) if args.tea_model2 else 0
        all_feat_snn_tensor = torch.cat(all_feat_snn, dim=0) if args.tea_model3 else 0

        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_logits_labels_abmil_snn_tensor = torch.cat(all_logits_labels_abmil_snn, dim=0) if args.tea_model1 else 0
        all_logits_labels_abmil_tensor = torch.cat(all_logits_labels_abmil, dim=0) if args.tea_model2 else 0
        all_logits_labels_snn_tensor = torch.cat(all_logits_labels_snn, dim=0) if args.tea_model3 else 0

        if args.intermediate_loss_fn == 'SP' or args.intermediate_loss_fn == 'PKT':
        # loss_rkd_1 = inter_fn(all_afeat_tensor * batch_alpha.view(bsz, 1), all_feat_abmil_snn_tensor * batch_alpha.view(bsz, 1)) if args.tea_model1 else 0
        # loss_rkd_2 = inter_fn(all_afeat_tensor * batch_beta.view(bsz, 1), all_feat_abmil_tensor *  batch_beta.view(bsz, 1)) if args.tea_model2 else 0
        # loss_rkd_3 = inter_fn(all_afeat_tensor * batch_gamma.view(bsz, 1), all_feat_snn_tensor * batch_gamma.view(bsz, 1)) if args.tea_model3 else 0
            loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor) if args.tea_model1 else 0
            loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor) if args.tea_model2 else 0
            loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor) if args.tea_model3 else 0
            loss_rkd = (loss_rkd_1 + loss_rkd_2 + loss_rkd_3) * bsz
        elif args.intermediate_loss_fn == 'AB':
            loss_rkd1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor.detach()) if args.tea_model1 else 0
            loss_rkd2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor.detach()) if args.tea_model2 else 0
            loss_rkd3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor.detach()) if args.tea_model3 else 0
            loss_rkd = loss_rkd1 + loss_rkd2 + loss_rkd3
        elif args.intermediate_loss_fn == 'TDC':
            batch_indexs = torch.stack(indexs, dim=0).view(-1).to(device)
            batch_sample_idxs = torch.stack(sample_idxs, dim=0).to(device)
            batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
            student_preds_tensor = torch.stack(student_preds, dim=0).to(device).squeeze(1)
            teacher1_preds_tensor = torch.stack(teacher1_preds, dim=0).to(device).squeeze(1) if args.tea_model1 else 0
            teacher2_preds_tensor = torch.stack(teacher2_preds, dim=0).to(device).squeeze(1) if args.tea_model2 else 0
            teacher3_preds_tensor = torch.stack(teacher3_preds, dim=0).to(device).squeeze(1) if args.tea_model3 else 0
            teacher1_sample_weights = assign_sample_weights(student_preds_tensor, teacher1_preds_tensor, batch_labels, 1, 1) if args.tea_model1 else 0
            teacher2_sample_weights = assign_sample_weights(student_preds_tensor, teacher2_preds_tensor, batch_labels, 1, 1) if args.tea_model2 else 0
            teacher3_sample_weights = assign_sample_weights(student_preds_tensor, teacher3_preds_tensor, batch_labels, 1, 1) if args.tea_model3 else 0
            loss_rkd_1, sample_loss_rkd_1 = inter_fn[0](teacher1_sample_weights, all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
            loss_rkd_2, sample_loss_rkd_2 = inter_fn[1](teacher2_sample_weights, all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
            loss_rkd_3, sample_loss_rkd_3 = inter_fn[2](teacher3_sample_weights, all_afeat_tensor, all_feat_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        elif args.intermediate_loss_fn == 'CRD':
            batch_indexs = torch.stack(indexs, dim=0).view(-1).to(device)
            batch_sample_idxs = torch.stack(sample_idxs, dim=0).to(device)
            batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
            loss_rkd_1 = inter_fn[0](all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
            loss_rkd_2 = inter_fn[1](all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
            loss_rkd_3 = inter_fn[2](all_afeat_tensor, all_feat_snn_tensor.detach(), batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        elif args.intermediate_loss_fn == 'SP_M':
            loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor, all_logits_labels_abmil_snn_tensor) if args.tea_model1 else 0
            loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor, all_logits_labels_abmil_tensor) if args.tea_model2 else 0
            loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor, all_logits_labels_snn_tensor) if args.tea_model3 else 0
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        else:
            loss_rkd = 0

        if args.logits_loss_fn == 'PKT' or args.logits_loss_fn == 'KL'or args.logits_loss_fn == 'BKD' or args.logits_loss_fn == 'MLKD':
            kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach())  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach())  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach())  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        elif args.logits_loss_fn == 'DKD':
            batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
            kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels, args.temp)  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        elif args.logits_loss_fn == 'NKD':
            batch_labels = torch.stack(labels, dim=0).view(-1).to(device)
            kl_loss_1 = logtis_fn[0](all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels,)  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn[0](all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels)  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn[0](all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels)  if args.tea_model3 else 0
            kl_loss_1_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels,)  if args.tea_model1 else 0
            kl_loss_2_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model2 else 0
            kl_loss_3_1 = logtis_fn[1](all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_1_1 + kl_loss_2_1 + kl_loss_3_1
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        else:
            kl_loss = 0

        loss_kd = loss_rkd + kl_loss
        loss_kd.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update_ema_variables(model, tea_model_abmil, 0.999, iter_num)
        iter_num += 1
        all_afeat.clear()
        all_feat_abmil_snn.clear()
        all_feat_abmil.clear()
        all_feat_snn.clear()
        alphas.clear()
        betas.clear()
        gammas.clear()
        indexs.clear()
        sample_idxs.clear()
        labels.clear()
        student_preds.clear()
        teacher1_preds.clear()
        teacher2_preds.clear()
        teacher3_preds.clear()
        all_logits.clear()
        all_logits_labels_abmil_snn.clear()
        all_logits_labels_abmil.clear()
        all_logits_labels_snn.clear()
        train_loss += loss_kd.item()

    # train_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    scheduler.step()

    train_epoch_str = 'Epoch: {}, train_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f} BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, train_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)

all_val_afeat = []
all_val_feat_abmil_snn = []
all_val_feat_abmil = []
all_val_feat_snn = []

all_val_logits = []
all_val_logits_labels_abmil_snn = []
all_val_logits_labels_abmil = []
all_val_logits_labels_snn = []

val_alphas = []
val_betas = []
val_gammas = []

val_indexs = []
val_sample_idxs = []
val_labels = []

val_student_preds = []
val_teacher1_preds = []
val_teacher2_preds = []
val_teacher3_preds = []
def validate_classification_coattn(cur, epoch, model, tea_model_abmil_snn, tea_model_abmil, tea_model_snn, loader, train_class_idx, 
                                inter_fn, logtis_fn, AUROC, AP, metrics, early_stopping=None, writer=None, loss_fn=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)
    if isinstance(inter_fn, list):
        for inter_fns in inter_fn:
            inter_fns = inter_fns.to(device)
    elif isinstance(inter_fn, bool):
        pass
    else:
        inter_fn = inter_fn.to(device)
        
    if not callable(logtis_fn):
        if isinstance(logtis_fn, bool):
            pass
        else:
            logtis_fn = logtis_fn.to(device)
    tea_model_abmil_snn = tea_model_abmil_snn.to(device) if args.tea_model1 else None
    tea_model_abmil = tea_model_abmil.to(device) if args.tea_model2 else None
    tea_model_snn = tea_model_snn.to(device) if args.tea_model3 else None
    k = args.nce_k
    global all_val_afeat, all_val_feat_abmil_snn, all_val_feat_abmil, all_val_feat_snn
    global val_alphas, val_betas, val_gammas, val_indexs, val_sample_idxs, val_labels, val_student_preds, val_teacher1_preds, val_teacher2_preds, val_teacher3_preds
    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):
        assert batch_idx in  train_class_idx[label[0]]
        data_WSI = data_WSI.cuda()
        data_omic = data_omic.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()

        all_neg_idx = list(range(0, len(loader)))
        all_neg_idx.remove(index[0])
        replace = True if k > len(all_neg_idx) else False
        neg_idx = np.random.choice(all_neg_idx, k, replace=replace) # 选择8个负样本
        sample_idx = np.hstack([index, neg_idx])

        with torch.no_grad():
            logits, Y_prob, Y_hat, afeat= model(x_path=data_WSI, x_omic=data_omic)
            val_student_preds.append(F.softmax(logits, dim=0))
            if args.tea_model1:
                logits_labels_abmil_snn, abmil_snn_prob, _, feat_abmil_snn= tea_model_abmil_snn(x_path=data_WSI, x_omic=data_omic)
                alpha = F.softmax(torch.cat([logits[0], logits_labels_abmil_snn[0]], dim=0)/args.temp, dim=0)
                alpha = alpha[2 + label.item()]/alpha[label.item()]
                val_teacher1_preds.append(F.softmax(logits, dim=1))
            else:
                alpha, logits_labels_abmil_snn, feat_abmil_snn = 0, 0, 0
            if args.tea_model2:
                logits_labels_abmil, abmil_prob, _, feat_abmil = tea_model_abmil(x_path=data_WSI)
                beta = F.softmax(torch.cat([logits[0], logits_labels_abmil[0]], dim=0)/args.temp, dim=0)
                beta = beta[2 + label.item()]/beta[label.item()]
                val_teacher2_preds.append(F.softmax(logits_labels_abmil, dim=1))
            else:
                beta, logits_labels_abmil, feat_abmil = 0, 0, 0
            if args.tea_model3:
                logits_labels_snn, snn_prob, _, feat_snn = tea_model_snn(feat=data_omic) 
                gamma = F.softmax(torch.cat([logits[0], logits_labels_snn[0]], dim=0)/args.temp, dim=0)
                gamma = gamma[2 + label.item()]/gamma[label.item()]
                val_teacher3_preds.append(F.softmax(logits_labels_snn, dim=1))
            else:
                gamma, logits_labels_snn, feat_snn = 0, 0, 0

            
        val_alphas.append(alpha)
        val_betas.append(beta)
        val_gammas.append(gamma)
        val_indexs.append(torch.tensor(index))
        val_sample_idxs.append(torch.tensor(sample_idx))
        val_labels.append(label)

        loss_class = loss_fn(logits, label)
        # kl_loss_1 = logtis_fn(logits, logits_labels_abmil_snn.detach()) * alpha if args.tea_model1 else 0
        # kl_loss_2 = logtis_fn(logits, logits_labels_abmil.detach()) * beta if args.tea_model2 else 0
        # kl_loss3 = logtis_fn(logits, logits_labels_snn.detach())  * gamma if args.tea_model3 else 0
        # kl_loss_1 = logtis_fn(logits, logits_labels_abmil_snn.detach())  if args.tea_model1 else 0
        # kl_loss_2 = logtis_fn(logits, logits_labels_abmil.detach())  if args.tea_model2 else 0
        # kl_loss3 = logtis_fn(logits, logits_labels_snn.detach())  if args.tea_model3 else 0
        # kl_loss = kl_loss_1 + kl_loss_2 + kl_loss3


        all_val_afeat.append(afeat)
        all_val_feat_abmil_snn.append(feat_abmil_snn)
        all_val_feat_abmil.append(feat_abmil)
        all_val_feat_snn.append(feat_snn)

        all_val_logits.append(logits)
        all_val_logits_labels_abmil_snn.append(logits_labels_abmil_snn)
        all_val_logits_labels_abmil.append(logits_labels_abmil)
        all_val_logits_labels_snn.append(logits_labels_snn)

        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss = loss / args.gc
        val_loss += loss.item()

        if (batch_idx + 1) % args.gc == 0: 
            val_batch_alpha = torch.stack(val_alphas, dim=0) if args.tea_model1 else 0
            val_batch_beta = torch.stack(val_betas, dim=0) if args.tea_model2 else 0
            val_batch_gamma = torch.stack(val_gammas, dim=0) if args.tea_model3 else 0
            bsz = val_batch_gamma.size(0)

            all_afeat_tensor = torch.cat(all_val_afeat, dim=0)
            all_feat_abmil_snn_tensor = torch.cat(all_val_feat_abmil_snn, dim=0) if args.tea_model1 else 0
            all_feat_abmil_tensor = torch.cat(all_val_feat_abmil, dim=0) if args.tea_model2 else 0
            all_feat_snn_tensor = torch.cat(all_val_feat_snn, dim=0) if args.tea_model3 else 0

            all_logits_tensor = torch.cat(all_val_logits, dim=0)
            all_logits_labels_abmil_snn_tensor = torch.cat(all_val_logits_labels_abmil_snn, dim=0) if args.tea_model1 else 0
            all_logits_labels_abmil_tensor = torch.cat(all_val_logits_labels_abmil, dim=0) if args.tea_model2 else 0
            all_logits_labels_snn_tensor = torch.cat(all_val_logits_labels_snn, dim=0) if args.tea_model3 else 0

            if args.intermediate_loss_fn == 'SP' or args.intermediate_loss_fn == 'PKT':
                # loss_rkd_1 = inter_fn(all_afeat_tensor * val_batch_alpha.view(bsz, 1), all_feat_abmil_snn_tensor * val_batch_alpha.view(bsz, 1)) if args.tea_model1 else 0
                # loss_rkd_2 = inter_fn(all_afeat_tensor * val_batch_beta.view(bsz, 1), all_feat_abmil_tensor *  val_batch_beta.view(bsz, 1)) if args.tea_model2 else 0
                # loss_rkd_3 = inter_fn(all_afeat_tensor * val_batch_gamma.view(bsz, 1), all_feat_snn_tensor * val_batch_gamma.view(bsz, 1)) if args.tea_model3 else 0
                loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor) if args.tea_model1 else 0
                loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor) if args.tea_model2 else 0
                loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor) if args.tea_model3 else 0
                loss_rkd = (loss_rkd_1 + loss_rkd_2 + loss_rkd_3) * bsz
            elif args.intermediate_loss_fn == 'AB':
                loss_rkd1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor.detach()) if args.tea_model1 else 0
                loss_rkd2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor.detach()) if args.tea_model2 else 0
                loss_rkd3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor.detach()) if args.tea_model3 else 0
                loss_rkd = loss_rkd1 + loss_rkd2 + loss_rkd3
            elif args.intermediate_loss_fn == 'TDC':
                batch_indexs = torch.stack(val_indexs, dim=0).view(-1).to(device)
                batch_sample_idxs = torch.stack(val_sample_idxs, dim=0).to(device)
                batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
                val_student_preds_tensor = torch.stack(val_student_preds, dim=0).to(device).squeeze(1)
                val_teacher1_preds_tensor = torch.stack(val_teacher1_preds, dim=0).to(device).squeeze(1) if args.tea_model1 else 0
                val_teacher2_preds_tensor = torch.stack(val_teacher2_preds, dim=0).to(device).squeeze(1) if args.tea_model2 else 0
                val_teacher3_preds_tensor = torch.stack(val_teacher3_preds, dim=0).to(device).squeeze(1) if args.tea_model3 else 0
                teacher1_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher1_preds_tensor, batch_labels, 1, 1) if args.tea_model1 else 0
                teacher2_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher2_preds_tensor, batch_labels, 1, 1) if args.tea_model2 else 0
                teacher3_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher3_preds_tensor, batch_labels, 1, 1) if args.tea_model3 else 0

                loss_rkd_1, sample_loss_rkd_1 = inter_fn[0](teacher1_sample_weights, all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
                loss_rkd_2, sample_loss_rkd_2 = inter_fn[1](teacher2_sample_weights, all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
                loss_rkd_3, sample_loss_rkd_3 = inter_fn[2](teacher3_sample_weights, all_afeat_tensor, all_feat_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            elif args.intermediate_loss_fn == 'CRD':
                batch_indexs = torch.stack(val_indexs, dim=0).view(-1).to(device)
                batch_sample_idxs = torch.stack(val_sample_idxs, dim=0).to(device)
                batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
                loss_rkd_1 = inter_fn[0](all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
                loss_rkd_2 = inter_fn[1](all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
                loss_rkd_3 = inter_fn[2](all_afeat_tensor, all_feat_snn_tensor.detach(), batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            elif args.intermediate_loss_fn == 'SP_M':
                loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor, all_logits_labels_abmil_snn_tensor) if args.tea_model1 else 0
                loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor, all_logits_labels_abmil_tensor) if args.tea_model2 else 0
                loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor, all_logits_labels_snn_tensor) if args.tea_model3 else 0
                loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
            else:
                loss_rkd = 0
            
            if args.logits_loss_fn == 'PKT' or args.logits_loss_fn == 'KL'or args.logits_loss_fn == 'BKD' or args.logits_loss_fn == 'MLKD':
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach())  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach())  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach())  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
            elif args.logits_loss_fn == 'DKD':
                batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels, args.temp)  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
            elif args.logits_loss_fn == 'NKD':
                batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels,)  if args.tea_model1 else 0
                kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels)  if args.tea_model2 else 0
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels)  if args.tea_model3 else 0

                kl_loss_1_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model1 else 0
                kl_loss_2_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model2 else 0
                kl_loss_3_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model3 else 0
                kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_1_1 + kl_loss_2_1 + kl_loss_3_1
            else:
                kl_loss = 0

            loss_kd  = loss_rkd + kl_loss
            all_val_afeat.clear()
            all_val_feat_abmil_snn.clear()
            all_val_feat_abmil.clear()
            all_val_feat_snn.clear()
            val_alphas.clear()
            val_betas.clear()
            val_gammas.clear()
            val_indexs.clear()
            val_sample_idxs.clear()
            val_labels.clear()
            val_student_preds.clear()
            val_teacher1_preds.clear()
            val_teacher2_preds.clear()
            val_teacher3_preds.clear()
            all_val_logits.clear()
            all_val_logits_labels_abmil_snn.clear()
            all_val_logits_labels_abmil.clear()
            all_val_logits_labels_snn.clear()
            val_loss += loss_kd.item()

    # 检查是否有剩余的梯度需要更新
    if (batch_idx + 1) % args.gc != 0:
        val_batch_alpha = torch.stack(val_alphas, dim=0) if args.tea_model1 else 0
        val_batch_beta = torch.stack(val_betas, dim=0) if args.tea_model2 else 0
        val_batch_gamma = torch.stack(val_gammas, dim=0) if args.tea_model3 else 0
        bsz = val_batch_gamma.size(0)

        all_afeat_tensor = torch.cat(all_val_afeat, dim=0) 
        all_feat_abmil_snn_tensor = torch.cat(all_val_feat_abmil_snn, dim=0) if args.tea_model1 else 0
        all_feat_abmil_tensor = torch.cat(all_val_feat_abmil, dim=0) if args.tea_model2 else 0
        all_feat_snn_tensor = torch.cat(all_val_feat_snn, dim=0) if args.tea_model3 else 0

        all_logits_tensor = torch.cat(all_val_logits, dim=0)
        all_logits_labels_abmil_snn_tensor = torch.cat(all_val_logits_labels_abmil_snn, dim=0) if args.tea_model1 else 0
        all_logits_labels_abmil_tensor = torch.cat(all_val_logits_labels_abmil, dim=0) if args.tea_model2 else 0
        all_logits_labels_snn_tensor = torch.cat(all_val_logits_labels_snn, dim=0) if args.tea_model3 else 0

        if args.intermediate_loss_fn == 'SP' or args.intermediate_loss_fn == 'PKT':
            # loss_rkd_1 = inter_fn(all_afeat_tensor * val_batch_alpha.view(bsz, 1), all_feat_abmil_snn_tensor * val_batch_alpha.view(bsz, 1)) if args.tea_model1 else 0
            # loss_rkd_2 = inter_fn(all_afeat_tensor * val_batch_beta.view(bsz, 1), all_feat_abmil_tensor *  val_batch_beta.view(bsz, 1)) if args.tea_model2 else 0
            # loss_rkd_3 = inter_fn(all_afeat_tensor * val_batch_gamma.view(bsz, 1), all_feat_snn_tensor * val_batch_gamma.view(bsz, 1)) if args.tea_model3 else 0
            loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor) if args.tea_model1 else 0
            loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor) if args.tea_model2 else 0
            loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor) if args.tea_model3 else 0
            loss_rkd = (loss_rkd_1 + loss_rkd_2 + loss_rkd_3) * bsz
        elif args.intermediate_loss_fn == 'AB':
            loss_rkd1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor.detach()) if args.tea_model1 else 0
            loss_rkd2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor.detach()) if args.tea_model2 else 0
            loss_rkd3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor.detach()) if args.tea_model3 else 0
            loss_rkd = loss_rkd1 + loss_rkd2 + loss_rkd3
        elif args.intermediate_loss_fn == 'TDC':
            batch_indexs = torch.stack(val_indexs, dim=0).view(-1).to(device)
            batch_sample_idxs = torch.stack(val_sample_idxs, dim=0).to(device)
            batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
            val_student_preds_tensor = torch.stack(val_student_preds, dim=0).to(device).squeeze(1)
            val_teacher1_preds_tensor = torch.stack(val_teacher1_preds, dim=0).to(device).squeeze(1) if args.tea_model1 else 0
            val_teacher2_preds_tensor = torch.stack(val_teacher2_preds, dim=0).to(device).squeeze(1) if args.tea_model2 else 0
            val_teacher3_preds_tensor = torch.stack(val_teacher3_preds, dim=0).to(device).squeeze(1) if args.tea_model3 else 0
            teacher1_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher1_preds_tensor, batch_labels, 1, 1) if args.tea_model1 else 0
            teacher2_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher2_preds_tensor, batch_labels, 1, 1) if args.tea_model2 else 0
            teacher3_sample_weights = assign_sample_weights(val_student_preds_tensor, val_teacher3_preds_tensor, batch_labels, 1, 1) if args.tea_model3 else 0
            loss_rkd_1, sample_loss_rkd_1 = inter_fn[0](teacher1_sample_weights, all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
            loss_rkd_2, sample_loss_rkd_2 = inter_fn[1](teacher2_sample_weights, all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
            loss_rkd_3, sample_loss_rkd_3 = inter_fn[2](teacher3_sample_weights, all_afeat_tensor, all_feat_snn_tensor.detach(), batch_labels, batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        elif args.intermediate_loss_fn == 'CRD':
            batch_indexs = torch.stack(val_indexs, dim=0).view(-1).to(device)
            batch_sample_idxs = torch.stack(val_sample_idxs, dim=0).to(device)
            batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
            loss_rkd_1 = inter_fn[0](all_afeat_tensor, all_feat_abmil_snn_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model1 else (0,torch.zeros(bsz).to(device))
            loss_rkd_2 = inter_fn[1](all_afeat_tensor, all_feat_abmil_tensor.detach(), batch_indexs, batch_sample_idxs) if args.tea_model2 else (0,torch.zeros(bsz).to(device))
            loss_rkd_3 = inter_fn[2](all_afeat_tensor, all_feat_snn_tensor.detach(), batch_indexs, batch_sample_idxs)if args.tea_model3 else (0,torch.zeros(bsz).to(device))
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        elif args.intermediate_loss_fn == 'SP_M':
            loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor, all_logits_labels_abmil_snn_tensor) if args.tea_model1 else 0
            loss_rkd_2 = inter_fn(all_afeat_tensor, all_feat_abmil_tensor, all_logits_labels_abmil_tensor) if args.tea_model2 else 0
            loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor, all_logits_labels_snn_tensor) if args.tea_model3 else 0
            loss_rkd = loss_rkd_1 + loss_rkd_2 + loss_rkd_3
        else:
            loss_rkd = 0

        if args.logits_loss_fn == 'PKT' or args.logits_loss_fn == 'KL'or args.logits_loss_fn == 'BKD' or args.logits_loss_fn == 'MLKD':
            kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach())  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach())  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach())  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        elif args.logits_loss_fn == 'DKD':
            batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
            kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels, args.temp)  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels, args.temp)  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        elif args.logits_loss_fn == 'NKD':
            batch_labels = torch.stack(val_labels, dim=0).view(-1).to(device)
            kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach(), batch_labels,)  if args.tea_model1 else 0
            kl_loss_2 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_tensor.detach(), batch_labels)  if args.tea_model2 else 0
            kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach(), batch_labels)  if args.tea_model3 else 0

            kl_loss_1_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model1 else 0
            kl_loss_2_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model2 else 0
            kl_loss_3_1 = logtis_fn(all_afeat_tensor, all_logits_tensor, batch_labels)  if args.tea_model3 else 0
            kl_loss = kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_1_1 + kl_loss_2_1 + kl_loss_3_1
        else:
            kl_loss = 0

        loss_kd = loss_rkd + kl_loss
        all_val_afeat.clear()
        all_val_feat_abmil_snn.clear()
        all_val_feat_abmil.clear()
        all_val_feat_snn.clear()
        val_alphas.clear()
        val_betas.clear()
        val_gammas.clear()
        val_indexs.clear()
        val_sample_idxs.clear()
        val_labels.clear()
        val_student_preds.clear()
        val_teacher1_preds.clear()
        val_teacher2_preds.clear()
        val_teacher3_preds.clear()
        all_val_logits.clear()
        all_val_logits_labels_abmil_snn.clear()
        all_val_logits_labels_abmil.clear()
        all_val_logits_labels_snn.clear()
        val_loss += loss_kd.item()

    # val_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'Epoch: {}, val_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)

    if early_stopping:
        assert args.results_dir
        if args.train_mode == 'auc':
            early_stopping(epoch, auroc, model, ckpt_name=os.path.join(args.results_dir, "s_{}_max_auc_checkpoint.pt".format(cur)))
        elif args.train_mode == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(args.results_dir, "s_{}_min_loss_checkpoint.pt".format(cur)))
        else:
            raise ValueError("train_mode should be 'auc' or 'loss'")
        
        if early_stopping.early_stop:
            print("Early stopping")
            best_model = model
            return val_loss, early_stopping.best_score, ap, metrics, True

    return val_loss, early_stopping.best_score, ap, metrics, False


def test_classification_coattn(model, loader, AUROC, AP, metrics, writer=None, loss_fn=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    # slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        data_omic = data_omic.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()

        with torch.no_grad():
            logits, Y_prob, Y_hat, _= model(x_path=data_WSI, x_omic=data_omic, mode='student')



        loss_class = loss_fn(logits, label)
        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss_value = loss.item()
        val_loss += loss_value


    val_loss /= len(loader)
    auroc = AUROC.compute()
    ap = AP.compute()
    metrics = metrics.compute()
    val_epoch_str = 'test_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format( val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('test/loss', val_loss)


    return val_loss, auroc, ap, metrics, False