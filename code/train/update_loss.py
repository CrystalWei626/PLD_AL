import torch
import torch.nn as nn
from pylab import *
sys.path.append('data/tyc/code/train')
import train.ramps as ramps

def get_current_consistency_weight(epoch,consistency,consistency_rampup):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_variance(pred1, pred2,targets):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    loss_kl = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)
    exp_loss_kl = torch.exp(-loss_kl)
    pseudo_label_rect = targets * exp_loss_kl
    return pseudo_label_rect,loss_kl

def update_variance_loss(pred1, pred2, loss_origin):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    loss_kl = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)
    exp_loss_kl = torch.exp(-loss_kl)
    loss_rect = torch.mean(loss_origin * exp_loss_kl) + torch.mean(loss_kl)
    return loss_rect

def update_consistency_loss(pred1, pred2,pseudo,pseudo_rect,T,threshold=0.8):
    if pseudo:
        criterion = nn.CrossEntropyLoss(reduction='none')
        pseudo_label = torch.softmax(pred2.detach() / T, dim=1)
        max_probs, targets = torch.max(pseudo_label, dim=1)
        if pseudo_rect:
            loss_ce = criterion(pred1, targets)
            loss = update_variance_loss(pred1, pred2, loss_ce)
        else:
            mask = max_probs.ge(threshold).float()
            loss_ce = criterion(pred1, targets)
            loss = torch.mean(loss_ce * mask)
    else:
        criterion = nn.MSELoss(reduction='none')
        loss_mse = criterion(pred1, pred2)
        loss = torch.mean(loss_mse)

    return loss
