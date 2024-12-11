import numpy as np
import torch
import os
import torch.nn.functional as F
import json
from utils.MMD_loss import MMD_Loss
import torchmetrics


all_afeat = []
all_feat_abmil_snn = []
all_feat_snn = []

all_logits = []
all_logits_labels_abmil_snn = []
all_logits_labels_snn = []

def train_loop_classification_coattn(epoch, model, tea_model_abmil_snn, tea_model_snn, loader, optimizer, scheduler, 
                                    inter_fn, logtis_fn, AUROC, AP, metrics, loss_fn=None, args=None):   
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    train_loss = 0.
    model = model.to(device)
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)
    inter_fn = inter_fn.to(device)
    logtis_fn = logtis_fn.to(device)
    # if not callable(inter_fn):
    #     if isinstance(inter_fn, list):
    #         for inter_fns in inter_fn:
    #             inter_fns = inter_fns.to(device)
    #     elif isinstance(inter_fn, bool):
    #         pass
    #     else:
    #         inter_fn = inter_fn.to(device)

    # if not callable(logtis_fn):
    #     if isinstance(logtis_fn, list):
    #         for logtis_fns in logtis_fn:
    #             logtis_fns = logtis_fns.to(device)
    #     elif isinstance(logtis_fn, bool):
    #         pass
    #     else:
    #         logtis_fn = logtis_fn.to(device)

    tea_model_abmil_snn = tea_model_abmil_snn.to(device)
    tea_model_snn = tea_model_snn.to(device)
    print('\n')
    global all_afeat, all_feat_abmil_snn, all_feat_snn, all_logits, all_logits_labels_abmil_snn, all_logits_labels_snn
    align_fn = MMD_Loss(kernel_type = 'mean_cov')
    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):


        label = label.type(torch.LongTensor).to(device)
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.type(torch.FloatTensor).to(device)
        logits, Y_prob, Y_hat, afeat= model(x_path=data_WSI)
        # student_preds.append(F.softmax(logits, dim=0))

        logits_labels_abmil_snn, _, _, afeat_abmil_snn= tea_model_abmil_snn(x_path=data_WSI, x_omic=data_omic)
        # teacher1_preds.append(F.softmax(logits_labels_abmil_snn, dim=1))

        logits_labels_snn, _, _, feat_snn = tea_model_snn(feat=data_omic) 
        # teacher3_preds.append(F.softmax(logits_labels_snn, dim=1))
        # aa = spearmanr(afeat.squeeze().cpu().detach().numpy(), afeat_abmil_snn.squeeze().cpu().detach().numpy())[0]
        # KRC_cors1.append(spearmanr(afeat.squeeze().cpu().detach().numpy(), afeat_abmil_snn.squeeze().cpu().detach().numpy())[0] > w)


        # cc = spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_snn.squeeze().cpu().detach().numpy())[0]
        # KRC_cors3.append(spearmanr(afeat.squeeze().cpu().detach().numpy(), feat_snn.squeeze().cpu().detach().numpy())[0] > w)




        loss_class = loss_fn(logits, label) + loss_fn(logits_labels_abmil_snn, label) + loss_fn(logits_labels_snn, label)
        loss = loss_class


        all_afeat.append(afeat)
        all_feat_abmil_snn.append(afeat_abmil_snn)
        all_feat_snn.append(feat_snn)
        all_logits.append(logits)
        all_logits_labels_abmil_snn.append(logits_labels_abmil_snn)
        all_logits_labels_snn.append(logits_labels_snn)


        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss / args.gc
        train_loss += loss.item()
        loss.backward(retain_graph=True)

        if (batch_idx + 1) % args.gc == 0: 
            
            bsz = args.gc
            all_afeat_tensor = torch.cat(all_afeat, dim=0)
            all_feat_abmil_snn_tensor = torch.cat(all_feat_abmil_snn, dim=0) 
            all_feat_snn_tensor = torch.cat(all_feat_snn, dim=0) 

            all_logits_tensor = torch.cat(all_logits, dim=0)
            all_logits_labels_abmil_snn_tensor = torch.cat(all_logits_labels_abmil_snn, dim=0) 
            all_logits_labels_snn_tensor = torch.cat(all_logits_labels_snn, dim=0)

            if args.intermediate_loss_fn == 'SP':
                loss_rkd_1 = inter_fn(all_afeat_tensor, all_feat_abmil_snn_tensor)
                loss_rkd_3 = inter_fn(all_afeat_tensor, all_feat_snn_tensor)
                loss_rkd = (loss_rkd_1 + loss_rkd_3) * bsz
            else:
                loss_rkd = 0
            


            if  args.logits_loss_fn == 'KL':
                kl_loss_1 = logtis_fn(all_logits_tensor, all_logits_labels_abmil_snn_tensor.detach()) + logtis_fn(all_logits_labels_abmil_snn_tensor, all_logits_tensor.detach())
                kl_loss_3 = logtis_fn(all_logits_tensor, all_logits_labels_snn_tensor.detach()) + logtis_fn(all_logits_labels_snn_tensor, all_logits_tensor.detach())
                kl_loss = kl_loss_1 + kl_loss_3
            else:
                kl_loss = 0

            if args.OR == 'OR':
                loss_ortho = (torch.abs(torch.mean(torch.sum(all_feat_abmil_snn_tensor * all_afeat_tensor, dim=1))) + \
                    torch.abs(torch.mean( torch.sum(all_feat_snn_tensor * all_afeat_tensor, dim=1))) + \
                        torch.abs(torch.mean( torch.sum(all_feat_snn_tensor * all_feat_abmil_snn_tensor, dim=1))))/6 
                # cos_sim_abmil_snn_afeat = F.cosine_similarity(all_feat_abmil_snn_tensor, all_afeat_tensor, dim=1)
                # cos_sim_snn_afeat = F.cosine_similarity(all_feat_snn_tensor, all_afeat_tensor, dim=1)
                # cos_sim_snn_abmil_snn = F.cosine_similarity(all_feat_snn_tensor, all_feat_abmil_snn_tensor, dim=1)
                # loss_ortho = (torch.abs(torch.mean(cos_sim_abmil_snn_afeat)) + 
                #             torch.abs(torch.mean(cos_sim_snn_afeat)) + 
                #             torch.abs(torch.mean(cos_sim_snn_abmil_snn))) * 6
                align_loss = align_fn(all_feat_abmil_snn_tensor, all_afeat_tensor) + align_fn(all_feat_snn_tensor, all_afeat_tensor) +  align_fn(all_feat_snn_tensor, all_feat_abmil_snn_tensor)
            else:
                loss_ortho = 0
                align_loss = 0
            loss_kd = loss_rkd + kl_loss + align_loss + loss_ortho
            loss_kd.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_afeat.clear()
            all_feat_abmil_snn.clear()
            all_feat_snn.clear()
            all_logits.clear()
            all_logits_labels_abmil_snn.clear()
            all_logits_labels_snn.clear()
            train_loss += loss_kd.item()

        # 检查是否有剩余的梯度需要更新
    if (batch_idx + 1) % args.gc != 0:
        # optimizer.step()
        optimizer.zero_grad()
        all_afeat.clear()
        all_feat_abmil_snn.clear()
        all_feat_snn.clear()
        all_logits.clear()
        all_logits_labels_abmil_snn.clear()
        all_logits_labels_snn.clear()
        train_loss += loss_kd.item()

    torch.cuda.empty_cache()
    train_loss /= len(loader) 
    train_loss *= args.gc
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


def validate_classification_coattn(cur, epoch, model, tea_model_abmil_snn, tea_model_snn, loader, 
                                AUROC, AP, metrics, early_stopping=None, loss_fn=None, args=None):
    model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loss = 0.
    model = model.to(device)
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)

    tea_model_abmil_snn = tea_model_abmil_snn.to(device) 
    tea_model_snn = tea_model_snn.to(device)

    snn_auroc = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    snn_auroc = snn_auroc.to(device)
    mul_auroc = torchmetrics.AUROC(task='binary', num_classes=args.n_classes, average = 'macro')
    mul_auroc = mul_auroc.to(device)

    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        data_omic = data_omic.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, afeat= model(x_path=data_WSI, x_omic=data_omic)
            logits_snn, Y_prob_snn, Y_hat_snn, _ = tea_model_snn(feat=data_omic)
            logits_mul, Y_prob_mul, Y_hat_mul, _ = tea_model_abmil_snn(x_path=data_WSI, x_omic=data_omic)

        loss_class = loss_fn(logits, label)
        AUROC.update(Y_prob[:,1], label.squeeze())
        snn_auroc.update(Y_prob_snn[:,1], label.squeeze())
        mul_auroc.update(Y_prob_mul[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss = loss / args.gc
        val_loss += loss.item()

    torch.cuda.empty_cache()

    val_loss /= len(loader)
    val_loss *= args.gc
    auroc = AUROC.compute()
    snn_auroc = snn_auroc.compute()
    mul_auroc = mul_auroc.compute()
    all_auroc = (auroc + snn_auroc + mul_auroc)/3
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'Epoch: {}, val_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')

    if early_stopping:
        assert args.results_dir
        if args.train_mode == 'auc':
            early_stopping(epoch, all_auroc, model, ckpt_name=os.path.join(args.results_dir, "s_{}.pt".format(cur)))
        elif args.train_mode == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(args.results_dir, "s_{}.pt".format(cur)))
        else:
            raise ValueError("train_mode should be 'auc' or 'loss'")
        
        if early_stopping.early_stop:
            print("Early stopping")
            # best_model = nn.ModuleList([model, tea_model_abmil_snn, tea_model_snn])
            if args.train_mode == 'auc': 
                return val_loss, early_stopping.best_score, ap, metrics, tea_model_abmil_snn, tea_model_snn, True
            else:
                return early_stopping.best_score, auroc, ap, metrics, tea_model_abmil_snn, tea_model_snn, True
    # best_model = nn.ModuleList([model, tea_model_abmil_snn, tea_model_snn])
    if args.train_mode == 'auc': 
        return val_loss, early_stopping.best_score, ap, metrics, tea_model_abmil_snn, tea_model_snn, False
    else:
        return early_stopping.best_score, auroc, ap, metrics, tea_model_abmil_snn, tea_model_snn, False



def test_classification_coattn(model, tea_snn, tea_mul, loader, all_auc, all_metrics, all_AP, loss_fn=None, args=None):
    model.eval()
    tea_snn.eval()
    tea_mul.eval()
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    val_loss = 0.
    global all_path_feat, all_omic_feat, all_mul_feat, zz
    for AUC in all_auc:
        AUC.to(device)
    for metrics in all_metrics:
        metrics.to(device)
    for AP in all_AP:
        AP.to(device)
    tea_snn = tea_snn.to(device)
    tea_mul = tea_mul.to(device)


    for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic = data_omic.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

        with torch.no_grad():
            logits, Y_prob, Y_hat, _= model(x_path=data_WSI, x_omic=data_omic)
            logits_snn, Y_prob_snn, Y_hat_snn, _ = tea_snn(feat=data_omic)
            logits_mul, Y_prob_mul, Y_hat_mul, _ = tea_mul(x_path=data_WSI, x_omic=data_omic)

        loss_class = loss_fn(logits, label) + loss_fn(logits_snn, label) + loss_fn(logits_mul, label)

        all_auc[0].update(Y_prob[:,1], label.squeeze())
        all_auc[1].update(Y_prob_snn[:,1], label.squeeze())
        all_auc[2].update(Y_prob_mul[:,1], label.squeeze())

        all_metrics[0].update(Y_hat, label)
        all_metrics[1].update(Y_hat_snn, label)
        all_metrics[2].update(Y_hat_mul, label)

        all_AP[0].update(Y_prob[:,1], label)
        all_AP[1].update(Y_prob_snn[:,1], label)
        all_AP[2].update(Y_prob_mul[:,1], label)

        loss = loss_class
        loss_value = loss.item()
        val_loss += loss_value


    val_loss /= len(loader)

    path_auc = all_auc[0].compute()
    omic_auc = all_auc[1].compute()
    mul_auc = all_auc[2].compute()


    path_metrics = all_metrics[0].compute()
    omic_metrics = all_metrics[1].compute()
    mul_metrics = all_metrics[2].compute()

    path_AP = all_AP[0].compute()
    omic_AP = all_AP[1].compute()
    mul_AP = all_AP[2].compute()


    metrics = {
        "Path Metrics": {},
        "Omic Metrics": {},
        "Mul Metrics": {}
    }

    for key, value in path_metrics.items():
        metrics["Path Metrics"][key] = value.item()
    metrics["Path Metrics"]['path_auc'] = path_auc.item()
    metrics["Path Metrics"]['path_ap'] = path_AP.item()

    for key, value in omic_metrics.items():
        metrics["Omic Metrics"][key] = value.item()
    metrics["Omic Metrics"]['omic_auc'] = omic_auc.item()
    metrics["Omic Metrics"]['omic_ap'] = omic_AP.item()

    for key, value in mul_metrics.items():
        metrics["Mul Metrics"][key] = value.item()
    metrics["Mul Metrics"]['mul_auc'] = mul_auc.item()
    metrics["Mul Metrics"]['mul_ap'] = mul_AP.item()

    # metrics = {k: tensor_to_serializable(v) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=4))

    return val_loss, metrics



# # visualize
# all_path_feat = []
# all_omic_feat = []
# all_mul_feat = []
# zz=0

# def test_classification_coattn(model, tea_snn, tea_mul, loader, all_auc, all_metrics, all_AP, loss_fn=None, args=None):
#     model.eval()
#     tea_snn.eval()
#     tea_mul.eval()
#     device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     val_loss = 0.
#     global all_path_feat, all_omic_feat, all_mul_feat, zz
#     for AUC in all_auc:
#         AUC.to(device)
#     for metrics in all_metrics:
#         metrics.to(device)
#     for AP in all_AP:
#         AP.to(device)
#     tea_snn = tea_snn.to(device)
#     tea_mul = tea_mul.to(device)


#     for batch_idx, (data_WSI, data_omic, label, index) in enumerate(loader):

#         data_WSI = data_WSI.to(device)
#         data_omic = data_omic.type(torch.FloatTensor).to(device)
#         label = label.type(torch.LongTensor).to(device)

#         with torch.no_grad():
#             logits, Y_prob, Y_hat, path_feat= model(x_path=data_WSI, x_omic=data_omic)
#             logits_snn, Y_prob_snn, Y_hat_snn, omic_feat = tea_snn(feat=data_omic)
#             logits_mul, Y_prob_mul, Y_hat_mul, mul_feat = tea_mul(x_path=data_WSI, x_omic=data_omic)
#             all_path_feat.append(path_feat.cpu())
#             all_omic_feat.append(omic_feat.cpu())
#             all_mul_feat.append(mul_feat.cpu())

#         loss_class = loss_fn(logits, label) + loss_fn(logits_snn, label) + loss_fn(logits_mul, label)

#         all_auc[0].update(Y_prob[:,1], label.squeeze())
#         all_auc[1].update(Y_prob_snn[:,1], label.squeeze())
#         all_auc[2].update(Y_prob_mul[:,1], label.squeeze())

#         all_metrics[0].update(Y_hat, label)
#         all_metrics[1].update(Y_hat_snn, label)
#         all_metrics[2].update(Y_hat_mul, label)

#         all_AP[0].update(Y_prob[:,1], label)
#         all_AP[1].update(Y_prob_snn[:,1], label)
#         all_AP[2].update(Y_prob_mul[:,1], label)

#         loss = loss_class
#         loss_value = loss.item()
#         val_loss += loss_value


#     val_loss /= len(loader)

#     path_auc = all_auc[0].compute()
#     omic_auc = all_auc[1].compute()
#     mul_auc = all_auc[2].compute()


#     path_metrics = all_metrics[0].compute()
#     omic_metrics = all_metrics[1].compute()
#     mul_metrics = all_metrics[2].compute()

#     path_AP = all_AP[0].compute()
#     omic_AP = all_AP[1].compute()
#     mul_AP = all_AP[2].compute()


#     metrics = {
#         "Path Metrics": {},
#         "Omic Metrics": {},
#         "Mul Metrics": {}
#     }

#     for key, value in path_metrics.items():
#         metrics["Path Metrics"][key] = value.item()
#     metrics["Path Metrics"]['path_auc'] = path_auc.item()
#     metrics["Path Metrics"]['path_ap'] = path_AP.item()

#     for key, value in omic_metrics.items():
#         metrics["Omic Metrics"][key] = value.item()
#     metrics["Omic Metrics"]['omic_auc'] = omic_auc.item()
#     metrics["Omic Metrics"]['omic_ap'] = omic_AP.item()

#     for key, value in mul_metrics.items():
#         metrics["Mul Metrics"][key] = value.item()
#     metrics["Mul Metrics"]['mul_auc'] = mul_auc.item()
#     metrics["Mul Metrics"]['mul_ap'] = mul_AP.item()

#     # metrics = {k: tensor_to_serializable(v) for k, v in metrics.items()}
#     print(json.dumps(metrics, indent=4))


#     data = {
#     'all_path_feat': [tensor.tolist() for tensor in all_path_feat],
#     'all_omic_feat': [tensor.tolist() for tensor in all_omic_feat],
#     'all_mul_feat': [tensor.tolist() for tensor in all_mul_feat],
#     }
#     zz+=1
#     files_name = f'data_{zz}.json'
#     with open(files_name, 'w') as file:
#         json.dump(data, file)
#     return val_loss, metrics


