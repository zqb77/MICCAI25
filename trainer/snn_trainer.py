import numpy as np
import torch
import os



def train_loop_classification_coattn(epoch, model, loader, optimizer, scheduler, AUROC, AP, metrics, writer=None, loss_fn=None, gc=16, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    shuffle = True
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)
    print('\n')
    
    for batch_idx, (feat, label) in enumerate(loader):

        
        feat = feat.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        logits, Y_prob, Y_hat, _ = model(x_omic = feat)
        loss_class = loss_fn(logits, label)

        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss_value = loss.item()
        train_loss += loss_value

        # loss = loss / gc + loss_reg
        loss = loss / gc
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    

    train_loss /= len(loader)
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


def validate_classification_coattn(cur, epoch, model, loader, AUROC, AP, metrics, early_stopping=None, writer=None, loss_fn=None, results_dir=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)

    for batch_idx, (feat, label) in enumerate(loader):

        feat = feat.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

        with torch.no_grad():
            logits, Y_prob, Y_hat, _= model(x_omic=feat)

        loss_class = loss_fn(logits, label)

        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss_value = loss.item()
        val_loss += loss_value


    val_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'Epoch: {}, val_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}'\
        .format(epoch, val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        # writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        # writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, auroc, model, ckpt_name=os.path.join(results_dir, "s_{}_max_auc_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            best_model = model
            return val_loss, auroc, ap, metrics, True

    return val_loss, auroc, ap, metrics, False


def test_classification_coattn(model, loader, AUROC, metrics, AP,  writer=None, loss_fn=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    val_loss = 0.
    metrics = metrics.to(device)
    AUROC = AUROC.to(device)
    AP = AP.to(device)
    # slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (feat, label) in enumerate(loader):

        feat = feat.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

        with torch.no_grad():
            logits, Y_prob, Y_hat, _ = model(x_omic=feat)
        loss_class = loss_fn(logits, label)

        AUROC.update(Y_prob[:,1], label.squeeze())
        metrics.update(Y_hat, label)
        AP.update(Y_prob[:,1], label)
        loss = loss_class
        loss_value = loss.item()

        val_loss += loss_value

    val_loss /= len(loader)
    auroc = AUROC.compute()
    metrics = metrics.compute()
    ap = AP.compute()
    val_epoch_str = 'test_loss: {:.4f}, auc: {:.4f}, ap: {:.4f}, BinaryAccuracy: {:.4f}, BinaryPrecision: {:.4f}, BinaryRecall: {:.4f}, BinarySpecificity: {:.4f}, BinaryF1Score: {:.4f}, BinaryCohensKappa: {:.4f}'\
        .format( val_loss, auroc, ap, metrics['BinaryAccuracy'], metrics['BinaryPrecision'], \
                metrics['BinaryRecall'], metrics['BinarySpecificity'], metrics['BinaryF1Score'], metrics['BinaryCohenKappa'])
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        # writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('test/loss', val_loss)
        # writer.add_scalar('val/c-index', c_index, epoch)


    return val_loss, auroc, ap, metrics, False