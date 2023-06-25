import os.path
import logging
import shutil
import math
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import h5py
import cleanlab
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.losses import dice,soft_ce,focal,jaccard
import sys
from medpy import metric
from scipy.ndimage import zoom
import random
from PIL import Image
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import pickle
from scipy.cluster.vq import *
from numpy import *
from PIL import Image
from pylab import *
from numpy import array,dot
from sklearn.cluster import KMeans
import pandas as pd
from itertools import cycle
import cv2

sys.path.append('D:/pythonProject/code')
from la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import losses
import ramps
import query_stg
import surface_distance as surfdist
from unet_mismatch import VNetMisMatchEfficient,VNetMisMatch
torch.cuda.device_count()
torch.cuda.is_available()
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

#设置参数
#root_path =  'D:/BaiduNetdiskDownload/Img_binary/h5_files/labeled_Aug'
root_path = 'D:\\BaiduNetdiskDownload\\data_CUBS\\h5_files_micii'
num_classes = 2
max_iterations = 1500
max_iterations_last = 1000
Max_AL_iter = 5  #主动学习最多循环次数
label_size = 200 #每次增加的标签数量
batch_size = 14
labeled_bs = 2
batch_size_test = 6
base_lr = 0.01
seed = 1000
ema = True
consistency_type = 'mse'
consistency = 1.0
consistency_rampup = 40.0
ema_decay = 0.99
beat_sup_metrics = []
#patch_size = [80,112]
patch_size = [128,256]
Loss = []
T = 1 #向前传播次数
threshold_pseudo = 0.9 #伪标签的阈值
K1 = 1
pseudo_rect = True #是否用EMA模型修正伪标签
pseudo = True
#correct_save_path = "D:/BaiduNetdiskDownload/Img_binary/h5_files/saved_correct/"
correct_save_path = os.path.join(root_path,'saved_correct\\')

# 损失函数
Loss_ce = soft_ce.SoftCrossEntropyLoss(smooth_factor=0.1, dim=1)
#Loss_focal = focal.FocalLoss(mode='binary')
Loss_focal = losses.FocalLoss()
Loss_dice = dice.DiceLoss(mode='binary')
Loss_jaccard = jaccard.JaccardLoss(mode='binary')
if consistency_type == 'mse':
    consistency_criterion = losses.softmax_mse_loss
elif consistency_type == 'kl':
    consistency_criterion = losses.softmax_kl_loss
else:
    assert False, consistency_type
if torch.cuda.is_available():
    Loss_ce = Loss_ce.cuda()
    Loss_focal = Loss_focal.cuda()
    Loss_dice = Loss_dice.cuda()
    Loss_jaccard = Loss_jaccard.cuda()

def write_in_h5(h5_save_path,images,labels,label_rect):
    with h5py.File(h5_save_path, mode='w') as hf:
        hf.create_dataset(name="image", dtype=float, data=images)
        hf.create_dataset(name="label", dtype=float, data=np.array(labels).astype(bool).astype(int))
        hf.create_dataset(name="label_rect", dtype=float, data=np.array(label_rect).astype(bool).astype(int))
    print(h5_save_path)

def get_data(data_path):
        h5f = h5py.File(data_path)
        image = Image.fromarray(np.array(h5f['image'][()]).astype('uint8'))
        image = np.array(image).reshape(1, image.size[1], image.size[0]).astype(np.float32)
        label = Image.fromarray(np.array(h5f['label'][()]).astype('uint8'))
        seg_label = Image.fromarray(np.array(h5f['label_rect'][()]).astype('uint8'))
        h5f.close()
        return torch.from_numpy(np.array(image)),torch.from_numpy(np.array(label)).long(),\
               torch.from_numpy(np.array(seg_label)).long()
#读取数据
class FullDataset(Dataset):
    def __init__(self, root_path,Init,type = 'Train', transform=None):
        self.transform = transform
        self.type = type
        if self.type == 'Train':
          self.label_dataset_dir = os.path.join(root_path,'train')
        if self.type == 'Test':
          self.label_dataset_dir = os.path.join(root_path,  'test')
        if self.type == 'Unlabel':
          self.label_dataset_dir = os.path.join(root_path, 'unlabeled')
        if self.type == 'Full_Train':
            self.label_dataset_dir = os.path.join(root_path, 'full_train')
        self.dir_list = os.listdir(self.label_dataset_dir)
        self.transform = transform
        self.data_path = []
        #self.pseudo_label_rect = []
        #self.KL_Dis = []
        self.Correct_Time = []
        if Init == True:
           self.init_seg_dict(self.type)

    def __getitem__(self, index):  # 读取数据和标签，并返回数据和标签
        #h5f = h5py.File(os.path.join(self.label_dataset_dir, self.dir_list[index]))
        #image = self.img_dict[index]
        #label = self.seg_dict[index]
        h5f = h5py.File(self.data_path[index])
        image = Image.fromarray(np.array(h5f['image'][()]).astype('uint8'))
        image = np.array(image).reshape(1, image.size[1], image.size[0]).astype(np.float32)
        label = Image.fromarray(np.array(h5f['label'][()]).astype('uint8'))
        label_rect = Image.fromarray(np.array(h5f['label_rect'][()]).astype('uint8'))
        sample = {'image': torch.from_numpy(np.array(image)),'label':torch.from_numpy(np.array(label)).long(),
                  'label_rect': torch.from_numpy(np.array(label_rect)).long()}
        h5f.close()
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # 必须写，返回数据集的长度
        return self.data_path.__len__()

    def init_seg_dict(self,type): #初始化
        for idx in range(len(self.dir_list)):
            self.Correct_Time.append(0)
            self.data_path.append(os.path.join(self.label_dataset_dir, self.dir_list[idx]))


    def Cauculate_KL(self):
        pred1 = self.prev_pred_prob_dict
        pred2 = self.prev_pred_prob_dict_ema
        length = pred1.__len__()
        pred_1 = pred1[0]
        pred_2 = pred2[0]
        for i in np.arange(1,length):
            pred_1 = torch.cat((pred_1,pred1[i]),dim=0)
            pred_2 = torch.cat((pred_2, pred1[i]), dim=0)
        pseudo_label = torch.softmax(pred_2.detach() / T, dim=1)
        max_probs, targets = torch.max(pseudo_label, dim=1)
        if pseudo_rect:    # 利用Model和EMA模型两个预测值的方差，对伪标签进行修正
            self.pseudo_label_rect,self.KL_Dis = update_variance(pred_1, pred_2, targets)
        else:
            mask = max_probs.ge(threshold_pseudo).float()  # 大于等于阈值
            self.pseudo_label_rect = targets * mask

    def update_dict(self,chosen,data_path):
        for c in chosen:
          self.data_path.append(data_path[c])
          self.Correct_Time.append(0)

    def delete_dict(self,chosen):
        chosen.sort()
        for index in chosen[::-1]:
          del self.data_path[index]

    # the core logic to update the pseudo annotation
    def update_seg_dict(self, idx,IoU_npl_indx,outputs_soft,out, mask_threshold=0.8):
        if self.Correct_Time[idx] <= 100:
              self.update_allclass(idx, IoU_npl_indx,outputs_soft,out, mask_threshold, 'single', class_constraint=True,
                                   update_or_mask='update', update_all_bg_img=True)
              self.Correct_Time[idx] += 1

    def update_allclass(self, idx, IoU_npl_indx,outputs_soft,out, mask_threshold, IoU_npl_constraint, class_constraint=True,
                        update_or_mask='update', update_all_bg_img=True):
        Img,label,seg_label = get_data(self.data_path[idx])  # h,w
        h, w = seg_label.size()  # h,w
        b = 1

        # if seg label does not belong to the set of class that needs to be updated (exclude the background class), return
        #if set(np.unique(seg_label.numpy())).isdisjoint(set(IoU_npl_indx[1:])):
            # only the background in the pseudo label
            # if update_all_bg_img and len(np.unique(seg_label.numpy()))==1 and np.unique(seg_label.numpy())[0]==0:
            #if update_all_bg_img and not (set(np.unique(seg_label.numpy()))-set(np.array([0,1]))):
            #    pass
            #else:
            #    return

        #seg_argmax, seg_prediction_prob = self.prev_pred_dict[idx],self.prev_pred_prob_dict[idx]
        #不能再用之前的字典了，现在来一个算一个！
        seg_argmax, seg_prediction_prob = out, outputs_soft
        seg_prediction_max_prob = seg_prediction_prob.max(axis = 1)[0]

        # if the class_constraint==True and seg label has foreground class
        # we prevent using predicted class that is not in the pseudo label to correct the label
        if class_constraint == True and (set(np.unique(seg_label[0].numpy())) - set(np.array([0, 255]))):
            for i_batch in range(b):
                unique_class = torch.unique(seg_label[i_batch])
                # print(unique_class)
                indx = torch.zeros((h, w), dtype=torch.long)
                for element in unique_class:
                    indx = indx | (seg_argmax[i_batch] == element)
                seg_argmax[i_batch][(indx == 0)] = 1

        #seg_mask_255 = (seg_argmax == 255) 我这里没有用255吧!

        # seg_change_indx means which pixels need to be updated,
        # find index where prediction is different from label,
        # and  it is not a ignored index and confidence is larger than threshold
        seg_label = seg_label.cpu()
        seg_argmax = seg_argmax.cpu()
        seg_prediction_max_prob = seg_prediction_max_prob.cpu()
        seg_change_indx = (seg_label != seg_argmax)  & (
                seg_prediction_max_prob > mask_threshold)   #这一步才是精华吧！！！

        # when set to "both", only when predicted class and pseudo label both existed in the set, the label would be corrected
        # this is a conservative way, during our whole experiments, IoU_npl_constraint is always set to be "single",
        # this is retained here in case user may find in useful for their dataset
        if IoU_npl_constraint == 'both':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
            class_indx_seg_label = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
                class_indx_seg_label = class_indx_seg_label | (seg_label == element)
            seg_change_indx = seg_change_indx & class_indx_seg_label & class_indx_seg_argmax

        #  when set to "single", when predicted class existed in the set, the label would be corrected, no need to consider pseudo label
        # e.g. when person belongs to the set, motor pixels in the pseudo label can be updated to person even if motor is not in set
        elif IoU_npl_constraint == 'single':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
            seg_change_indx = seg_change_indx & class_indx_seg_argmax

        # if the foreground class portion is too small, do not update
        seg_label_clone = seg_label.clone().unsqueeze(0).unsqueeze(0)
        seg_label_clone[seg_change_indx] = seg_argmax[seg_change_indx]
        if torch.sum(seg_label_clone!=0) < 0.5 * torch.sum(seg_label!=0) and torch.sum(seg_label_clone==0)/(b*h*w)>0.95:
            return

        # update or mask 255
        if update_or_mask == 'update':
            seg_label[seg_change_indx.squeeze(0).squeeze(0)] = seg_argmax[seg_change_indx]
            # update all class of the pseudo label
        else:
            # mask the pseudo label for 255 without computing the loss
            seg_label[seg_change_indx] = (torch.ones((b, h, w), dtype=torch.long) * 255)[
                seg_change_indx]  # the updated pseudo label

        #self.seg_dict[idx] = seg_label.cpu()
        #将校正后的label直接写入h5文件
        #write_in_h5(self.data_path[idx], Img.squeeze(0).detach().numpy(), label.detach().numpy(),seg_label.detach().numpy())
        h5f = h5py.File(self.data_path[idx], "r+")
        # h5f需要具有写权限，否则会提示“no write intent on file”错误
        h5f.__delitem__("label_rect")
        #h5f['image'] = Img.squeeze(0).detach().numpy()
        #h5f['label'] = label.detach().numpy()
        h5f['label_rect'] = seg_label.cpu().detach().numpy()
        h5f.close()
        pre = seg_label.cpu().detach().numpy() * 255
        cv2.imwrite(correct_save_path + str(idx) + '_correct_{}.png'.format(self.Correct_Time[idx]), pre.astype('int'))

#seed = 100
#def worker_init_fn(worker_id):
#    random.seed(seed+worker_id)

def DataLoad(root_path,Init):
#还需要数据增广！将图片剪切到相同的Size！！！不然后面 enumerate(DataLoader)会报错！！！
  train_dataset = FullDataset(root_path,Init,type='Train')#,transform=transforms.Compose([
                          #RandomRotFlip(),
                          #RandomCrop(patch_size),
                          #ToTensor()]))
  test_dataset = FullDataset(root_path,Init,type='Test')#, transform=transforms.Compose([
                  #RandomRotFlip(),
                  #RandomCrop(patch_size),
                  #ToTensor()]))
  unlabeled_dataset =FullDataset(root_path,Init,type='Unlabel')#,transform=transforms.Compose([
                           #RandomRotFlip(),
                           #CenterCrop(patch_size),
                           #ToTensor()]))
  return train_dataset,test_dataset,unlabeled_dataset

#关于半监督学习的函数
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def update_variance(pred1, pred2,targets):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    # 用loss_kl 近似等于 variance
    loss_kl = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)  # pred1 是student model, 被指导
    exp_loss_kl = torch.exp(-loss_kl)
    pseudo_label_rect = targets * exp_loss_kl  #后面那个Loss_kl正则项其实可以不用加了
    return pseudo_label_rect,loss_kl

def update_variance_loss(pred1, pred2, loss_origin):
    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    # 用loss_kl 近似等于 variance
    loss_kl = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)  # pred1 是student model, 被指导
    exp_loss_kl = torch.exp(-loss_kl)
    # print(variance.shape)
    # print('variance mean: %.4f' % torch.mean(exp_variance[:]))
    # print('variance min: %.4f' % torch.min(exp_variance[:]))
    # print('variance max: %.4f' % torch.max(exp_variance[:]))
    loss_rect = torch.mean(loss_origin * exp_loss_kl) + torch.mean(loss_kl)
    return loss_rect

def update_consistency_loss(pred1, pred2,threshold=0.8):
    if pseudo: #如果生成伪标签的话
        criterion = nn.CrossEntropyLoss(reduction='none')
        # 用pred2生成伪标签
        pseudo_label = torch.softmax(pred2.detach() / T, dim=1)  # T：向前传播次数（所以这里的伪标签生成并不是单纯地用这个Epoch的pred，而是用前几次的平均？）
        max_probs, targets = torch.max(pseudo_label, dim=1)    # 概率和标签下标
        # print(targets.shape)
        if pseudo_rect:    # 利用两个预测值的方差，对伪标签进行修正
            # Crossentropyloss作为损失函数时，iutput应该是[batchsize, n_class, h, w, d]，target是[batchsize, h, w, d]
            loss_ce = criterion(pred1, targets)  # 输出shape [batch, h, w, d]
            # print(pred1.shape, targets.shape)
            loss = update_variance_loss(pred1, pred2, loss_ce)
        else:
            mask = max_probs.ge(threshold).float()  # 大于等于阈值
            loss_ce = criterion(pred1, targets)
            loss = torch.mean(loss_ce * mask)
            # print(loss)
    else:
        criterion = nn.MSELoss(reduction='none')
        loss_mse = criterion(pred1, pred2)
        loss = torch.mean(loss_mse)

    return loss


#关于早期学习的函数
def update_iou_stat(predict, gt, TP, P, T, num_classes = 2):
    """
    :param predict: the pred of each batch,  should be numpy array, after take the argmax   b,h,w
    :param gt: the gt label of the batch, should be numpy array     b,h,w
    :param TP: True positive
    :param P: positive prediction
    :param T: True seg
    :param num_classes: number of classes in the dataset
    :return: TP, P, T
    """
    cal = gt < 255
    mask = (predict == gt) * cal
    for i in range(num_classes):
        P[i] += np.sum((predict == i) * cal)
        T[i] += np.sum((gt == i) * cal)
        TP[i] += np.sum((gt == i) * mask)

    return TP, P, T

def compute_iou(TP, P, T, num_classes = 2):
    """
    :param TP:
    :param P:
    :param T:
    :param num_classes: number of classes in the dataset
    :return: IoU
    """
    IoU = []
    for i in range(num_classes):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i] + 1e-10))
    return IoU

def compute_IOU(predict, gt,num_classes=2):
    TP = [0]*2
    T = [0]*2
    P = [0]*2
    cal = gt < 255
    mask = (predict == gt) * cal
    for i in range(num_classes):
        P[i] += np.sum((predict == i) * cal)
        T[i] += np.sum((gt == i) * cal)
        TP[i] += np.sum((gt == i) * mask)
    IoU = []
    for i in range(num_classes):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i] + 1e-10))
    #return np.mean(IoU)
    return IoU[0],IoU[1]


def curve_func(x, a, b, c): #拟合曲线
    return a * (1 - np.exp(-1 / c * x ** b))

def fit(func, x, y): #求解具有变量界限的线性 least-squares 问题。
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(y)),
                           absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]),maxfev = 10000)
    return tuple(popt)

def derivation(x, a, b, c):  #这个算的是导数，当导数变化超过一定阈值的时候校正标签
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))

def label_update_epoch(ydata_fit,  num_iter_per_epoch,current_epoch,threshold, eval_interval=100):
    #data_fit = np.linspace(0, len(ydata_fit) * eval_interval / num_iter_per_epoch, len(ydata_fit))
    #print(xdata_fit)
    xdata_fit = np.arange(1, current_epoch+2)
    #print(xdata_fit)
    a, b, c = fit(curve_func, xdata_fit, ydata_fit) #ydata_fit是IoU（度量模型性能），x应该表示训练时间，怎么度量呢？用Epoch？
    epoch = np.arange(1, current_epoch+2)
    #y_hat = curve_func(epoch, a, b, c) #表示拟合曲线在不同Epoch下的取值，属于预测？
    #relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c))) / abs(derivation(1, a, b, c))
    #relative_change[relative_change > 1] = 0
    abs_change = abs(abs(derivation(epoch, a, b, c))[current_epoch] - abs(derivation(epoch, a, b, c))[current_epoch-1])
    print(abs_change)
    #update_epoch = np.sum(relative_change <= threshold)
    if (abs_change <= threshold):
        update_epoch = True
    else:
        update_epoch = False
    return update_epoch,abs_change  # , a, b, c

def if_update(iou_value, current_epoch, num_iter_per_epoch,threshold):
    update_epoch,abs_change = label_update_epoch(iou_value,num_iter_per_epoch,current_epoch,threshold=threshold)
    return update_epoch,abs_change  # , update_epoch

def validation(test_loader,model):
    metric_dice_0 = 0.0
    metric_dice_1 = 0.0
    metric_iou_0 = 0.0
    metric_iou_1 = 0.0
    ASD = 0.0
    HD = 0.0
    for index in np.arange(test_loader.dataset.data_path.__len__()):
        input, label_test, _ = get_data(test_loader.dataset.data_path[index])  # Validation用的是精确标签！
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            outputs = outputs
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
            out = torch.argmax(outputs_soft, dim=1).squeeze(0).cpu()  # 预测的Label
            metric_dice_1 += np.sum(metric.binary.dc(out.numpy() == 1, np.array(label_test) == 1))
            # metric_dice_0 += np.sum(metric.binary.dc(out.numpy() == 0, np.array(label_test) == 0))
            metric_o, metric_o_1 = compute_IOU(out.numpy(), np.array(label_test))
            # metric_iou_0 += metric_o
            metric_iou_1 += metric_o_1

            surface_distances = surfdist.compute_surface_distances(
                label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                spacing_mm=(1.0, 1.0))
            avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
            ASD += avg_surf_dist[1]
            surface_distances = surfdist.compute_surface_distances(
                label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                spacing_mm=(1.0, 1.0))
            hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)  # 95HD
            HD += hd_dist_95
    # metric_dice_0 = metric_dice_0 / test_loader.dataset.data_path.__len__()
    metric_dice_1 = metric_dice_1 / test_loader.dataset.data_path.__len__()
    # metric_iou_0 = metric_iou_0 / test_loader.dataset.data_path.__len__()
    metric_iou_1 = metric_iou_1 / test_loader.dataset.data_path.__len__()
    ASD = ASD / test_loader.dataset.data_path.__len__()
    HD = HD / test_loader.dataset.data_path.__len__()

    return  metric_dice_1,metric_iou_1,ASD,HD


def Active_Learning_train(train_loader,unlabeled_loader,test_loader,AL_time,Rect):
    # use to record the updated class, so that it won't be updated again
    update_sign_list = []
    Updated_class_list = []
    Abs_change_1 = []
    Abs_change_0 = []
    # record the noisy pseudo label fitting IoU for each class
    IoU_npl_dict = {}
    for i in range(2):
        IoU_npl_dict[i] = []
    updated = False
    metric_list = 0.0
    # 损失函数
    Loss_ce = soft_ce.SoftCrossEntropyLoss(smooth_factor=0.1, dim=1)
    Loss_focal = focal.FocalLoss(mode='binary')
    Loss_dice = dice.DiceLoss(mode='binary')
    Loss_jaccard = jaccard.JaccardLoss(mode='binary')
    if consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, consistency_type
    if torch.cuda.is_available():
        Loss_ce = Loss_ce.cuda()
        Loss_focal = Loss_focal.cuda()
        Loss_dice = Loss_dice.cuda()
        Loss_jaccard = Loss_jaccard.cuda()
    # 构建模型
    params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        classes = 2
    )  # 如果给aux_params，则会输出分类辅助输出
    model_name = 'UNetPlusPlus'
    #之前：resnet34  imagenet
    model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    #preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    #global model
    #global ema_model
    ema_model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_depth=4,encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    for param in ema_model.parameters():  # 教师模型的结构和学生模型一样，但参数要通过指数移动平均更新
          param.detach_()
    ema_model = ema_model.cuda()
    model.cuda()
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_performance = 0.0
    if AL_time == (Max_AL_iter):
        max_iter = max_iterations_last
    else:
        max_iter = max_iterations
    max_epoch = max_iter // len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter(root_path + '/log')
    logging.basicConfig(filename=root_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    for epoch_num in iterator:
        # the noisy label fit IoU
        TP_npl_epoch = [0] * 2
        P_npl_epoch = [0] * 2
        T_npl_epoch = [0] * 2 #这个是计算每个Epoch的IoU
        for i_batch, sampled_batch in enumerate(zip(train_loader,unlabeled_loader)): #是不是不用Cycle会好些？
            img_batch = torch.cat((sampled_batch[0]['image'].to(torch.float32),sampled_batch[1]['image'].to(torch.float32)),
                                    dim=0)
            label_rect = sampled_batch[0]['label_rect']
            lab = sampled_batch[0]['label']
            if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              label_batch = label_rect.cuda().long()
            # 加入噪声
            noise = torch.clamp(torch.randn_like(img_batch) * 0.1, -0.2, 0.2)
            # student model + noise, teacher不加noise
            student_inputs = img_batch + noise
            ema_inputs = img_batch
            # 突然明白为什么输出的维度是【batch_size,num_classes,H,W】了！！！因为它输出的是每个Classes上的概率！我们要取的是更大的那个概率作为分类
            outputs = model(student_inputs)
            #outputs = outputs[0]
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
            out = torch.argmax(outputs_soft, dim=1).unsqueeze(1)  # 预测的Label
            #真正的伪标签要用这个Out去更新seg_dict
            with torch.no_grad():
                ema_output = ema_model(ema_inputs.cuda())
                #ema_output = ema_output[0]
            label_np_updated = label_batch.cpu().numpy()  # 这部分表示后面要根据早期学习更新的Label
            # 更新IoU
            #TP_npl, P_npl, T_npl = update_iou_stat(pred_np, label_np_updated, TP_npl,
            #                                       P_npl, T_npl)
            label_b = label_batch.shape[0]
            TP_npl_epoch, P_npl_epoch, T_npl_epoch = update_iou_stat(out[:label_b].squeeze(1).detach().cpu().numpy(),
                                                                     label_np_updated, TP_npl_epoch,
                                                                     P_npl_epoch, T_npl_epoch)
            # 损失函数里面有Softmax  监督损失
            loss_seg = F.cross_entropy(outputs[:label_b], label_batch[:label_b])
            loss_seg_dice = losses.dice_loss(outputs_soft[:label_b, 1, :, :], label_batch[:label_b] == 1) + \
                            losses.dice_loss(outputs_soft[:label_b, 0, :, :], label_batch[:label_b] == 0)
            supervised_loss = (loss_seg + 1.25*loss_seg_dice)

            # 扰动一致性损失，ramps up调整一致性损失在整个损失中的权重
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # 总损失（可能还会有权重什么的）
            consistency_loss = update_consistency_loss(outputs, ema_output)
            consistency_dist = consistency_weight * consistency_loss
            loss = supervised_loss + consistency_dist * 100
            #loss = supervised_loss #监督学习的话就只有supervised_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #这是控制梯度的大小二范数不超过20
            optimizer.step()
            # 更新EMA模型的参数
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            # 学习率衰减
            lr_ = base_lr * (1.0 - iter_num / (max_iter)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num +1
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_entropy: %f,loss_dice: %f' %
                         (iter_num, loss.item(), consistency_dist.item(),
                          loss_seg.item(),
                          loss_seg_dice.item()))

            #Loss.append([loss.item(),supervised_loss.item(),loss_seg_dice.item(),consistency_loss.item()])

            print('-' * 50)

            # Validation
            if iter_num > max_iterations/2 and iter_num % 100 == 0: #and AL_time == (Max_AL_iter):
                metric_dice_1,metric_iou_1,ASD,HD = validation(test_loader,ema_model)

                writer.add_scalar('info/val_dice',
                                  metric_list, iter_num)
                performance = metric_dice_1
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(root_path,
                                                  'AL_iter_{}_dice_{}_{}_{}_{}.pth'.format(
                                                      iter_num, round(best_performance, 4), round(metric_iou_1, 4),
                                                      round(ASD, 4),
                                                      round(HD, 4)))
                    save_best = os.path.join(root_path,
                                             'AL_{}_model_{}.pth'.format(AL_time, label_size))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))

                #model.train()
        #IoU_npl_epoch = compute_iou(TP_npl_epoch, P_npl_epoch, T_npl_epoch)
        #for i in range(2):  # 表示两个类（前景和背景都要更新）
        #    IoU_npl_dict[i].append(IoU_npl_epoch[i])
        #save_Time = os.path.join(root_path, 'AL_{}_Epoch_{}.pth'.format(AL_time, epoch_num))
        #torch.save(model.state_dict(), save_Time)
        #计算当前Epoch的平均IoU
        if AL_time != (Max_AL_iter) and Rect == True: #正常把+改成-
         IoU_npl_epoch = compute_iou(TP_npl_epoch,P_npl_epoch,T_npl_epoch)
         for i in range(2):  # 表示两个类（前景和背景都要更新）
            IoU_npl_dict[i].append(IoU_npl_epoch[i])
        # 计算当前网络的输出，存入prev_pred_dict里面
         if updated==False and epoch_num>=K1:
        # 每个Epoch更新标签
           IoU_npl_indx = [0]   # 这里要确保已经Updated的类别不能再Updated了
           update_sign_1,abs_change_1 = if_update(np.array(IoU_npl_dict[1]), epoch_num, len(train_loader),threshold=0.05)  # 前景标签的更新
           Abs_change_1.append(abs_change_1)
           update_sign_0,abs_change_0 = if_update(np.array(IoU_npl_dict[0]), epoch_num, len(train_loader), threshold=0.05)  # 背景标签的更新
           Abs_change_0.append(abs_change_0)
           update_sign_list.append([update_sign_0,update_sign_1])
           if update_sign_0 == True or update_sign_1==True:
              #在要优化的那一步计算标签数据的预测，节省内存
              for index in np.arange(train_loader.dataset.data_path.__len__()):
                   input,label,label_n = get_data(train_loader.dataset.data_path[index])
                   input = input.to(torch.float32) #c, h ,w
                   model.eval()
                   ema_model.eval()
                   with torch.no_grad():
                       outputs = model(input.unsqueeze(0).cuda())
                       #outputs = outputs[0]
                       ema_outputs = ema_model(input.unsqueeze(0).cuda())
                       #ema_outputs = ema_outputs[0]
                       outputs_soft = torch.softmax(ema_outputs, dim=1)  # 预测的概率分布  #注意是用ema模型的输出吗？？？
                       out = torch.argmax(outputs_soft, dim=1).unsqueeze(1).cpu()  # 预测的Label
                   if AL_time == 0: #Label和Img只保存一次，第一次的label_rect就是噪声标签
                       img = input.squeeze(0).squeeze(0).detach().cpu().numpy()
                       cv2.imwrite(correct_save_path + str(index) + '_image.png', img.astype('int'))
                       label_gt = label.squeeze(0).squeeze(0).detach().cpu().numpy()
                       lab = label_gt * 255
                       cv2.imwrite(correct_save_path + str(index) + '_label.png', lab.astype('int'))
                       label_noise = label_n.squeeze(0).squeeze(0).detach().cpu().numpy()
                       lab_noise = label_noise * 255
                       cv2.imwrite(correct_save_path + str(index) + '_label_noise.png', lab_noise.astype('int'))
                   if update_sign_0 == True and 0 not in Updated_class_list:
                     #Updated_class_list.append(0)
                     IoU_npl_indx = [0]
                     train_loader.dataset.update_seg_dict(index,IoU_npl_indx,outputs_soft,out,
                                                   mask_threshold=0.8)
                   if update_sign_1 == True and 1 not in Updated_class_list:
                     #Updated_class_list.append(1)
                     IoU_npl_indx = [1]
                     train_loader.dataset.update_seg_dict(index,IoU_npl_indx,outputs_soft,out,
                                                 mask_threshold=0.8)
              if update_sign_0 == True and 0 not in Updated_class_list:
                  Updated_class_list.append(0)
              if update_sign_1 == True and 1 not in Updated_class_list:
                  Updated_class_list.append(1)
           if 0 in Updated_class_list and 1 in Updated_class_list:
             updated = True  # 更新了一次就别再更新了 #为什么不能多Update几次啊？
             metric_dice_1, metric_iou_1, ASD, HD = validation(test_loader, ema_model)
             save_Time = os.path.join(root_path,'AL_{}_dice_{}_{}_{}_{}.pth'.format(AL_time,metric_dice_1,metric_iou_1,ASD,HD))
             torch.save(model.state_dict(), save_Time)
             print(Abs_change_0)
             print(Abs_change_1)
             f = open(os.path.join(root_path,"Abs_change_0_{}.txt".format(AL_time)), "w")
             f.writelines(str(Abs_change_0))
             f.close()
             f = open(os.path.join(root_path,"Abs_change_1_{}.txt".format(AL_time)), "w")
             f.writelines(str(Abs_change_1))
             f.close()
             #del model
             #del ema_model
             break
    beat_sup_metrics.append(best_performance)
    return model,ema_model

def Active_Learning_train_MeanTeacher(train_loader,unlabeled_loader,test_loader,AL_time,max_iterations = 3000):
    # 构建模型
    params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        classes = 2
    )  # 如果给aux_params，则会输出分类辅助输出
    model_name = 'UNetPlusPlus'
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    ema_model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    for param in ema_model.parameters():  # 教师模型的结构和学生模型一样，但参数要通过指数移动平均更新
        param.detach_()
    model.cuda()
    ema_model = ema_model.cuda()
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_performance = 0.0
    max_iter = max_iterations
    max_epoch = max_iter // len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter(root_path + '/log')
    logging.basicConfig(filename=root_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(train_loader,unlabeled_loader)): #是不是不用Cycle会好些？
            img_batch = torch.cat((sampled_batch[0]['image'].to(torch.float32),sampled_batch[1]['image'].to(torch.float32)),
                                    dim=0)
            unlabeled_volume_batch = sampled_batch[1]['image'].to(torch.float32)
            lab = sampled_batch[0]['label_rect']
            if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              label_batch = lab.cuda().long()
            # 加入噪声
            noise = torch.clamp(torch.randn_like(img_batch) * 0.1, -0.2, 0.2)
            # student model + noise, teacher不加noise
            student_inputs = img_batch + noise
            ema_inputs = img_batch
            # 突然明白为什么输出的维度是【batch_size,num_classes,H,W】了！！！因为它输出的是每个Classes上的概率！我们要取的是更大的那个概率作为分类
            outputs = model(student_inputs)
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
            out = torch.argmax(outputs_soft, dim=1).unsqueeze(1)  # 预测的Label
            #真正的伪标签要用这个Out去更新seg_dict
            with torch.no_grad():
                ema_output = ema_model(ema_inputs.cuda())
            label_b = label_batch.shape[0]
            # 损失函数里面有Softmax  监督损失
            loss_seg = F.cross_entropy(outputs[:label_b], label_batch[:label_b])
            loss_seg_dice = losses.dice_loss(outputs_soft[:label_b, 1, :, :], label_batch[:label_b] == 1) + \
                            losses.dice_loss(outputs_soft[:label_b, 0, :, :], label_batch[:label_b] == 0)
            supervised_loss = (loss_seg + 1.25*loss_seg_dice)

            # 计算不确定度
            T = 10  # 就是根据这10次的结果计算不确定度
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, patch_size[0], patch_size[1]]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs.cuda())
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, patch_size[0], patch_size[1])
            preds = torch.mean(preds, dim=0)  # (batch, 2, 112,80)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 112,80)
            # 扰动一致性损失，ramps up调整一致性损失在整个损失中的权重
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(outputs[label_b:], ema_output[label_b:])  # (batch, 2, 112,80)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist

            loss = supervised_loss + consistency_dist

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #这是控制梯度的大小二范数不超过20
            optimizer.step()
            # 更新EMA模型的参数
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            # 学习率衰减
            lr_ = base_lr * (1.0 - iter_num / (max_iter)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num +1
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_entropy: %f,loss_dice: %f' %
                         (iter_num, loss.item(), consistency_dist.item(),
                          loss_seg.item(),
                          loss_seg_dice.item()))

            #Loss.append([loss.item(),supervised_loss.item(),loss_seg_dice.item(),consistency_loss.item()])

            print('-' * 50)

            # Validation
            if iter_num > max_iterations/2 and iter_num % 200 == 0 and AL_time == (Max_AL_iter):
                metric_dice_0 = 0.0
                metric_dice_1 = 0.0
                metric_iou_0 = 0.0
                metric_iou_1 = 0.0
                ASD = 0.0
                HD = 0.0
                for index in np.arange(test_loader.dataset.data_path.__len__()):
                    input, label_test,_ = get_data(test_loader.dataset.data_path[index])  #Validation用的是精确标签！
                    input = input.to(torch.float32)  # c, h ,w
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input.unsqueeze(0).cuda())
                        outputs = outputs
                        outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
                        out = torch.argmax(outputs_soft, dim=1).squeeze(0).cpu()  # 预测的Label
                        metric_dice_1 += np.sum(metric.binary.dc(out.numpy() == 1, np.array(label_test) == 1))
                        #metric_dice_0 += np.sum(metric.binary.dc(out.numpy() == 0, np.array(label_test) == 0))
                        metric_o, metric_o_1 = compute_IOU(out.numpy(), np.array(label_test))
                        #metric_iou_0 += metric_o
                        metric_iou_1 += metric_o_1

                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                        ASD += avg_surf_dist[1]
                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)  # 95HD
                        HD += hd_dist_95
                #metric_dice_0 = metric_dice_0 / test_loader.dataset.data_path.__len__()
                metric_dice_1 = metric_dice_1 / test_loader.dataset.data_path.__len__()
                #metric_iou_0 = metric_iou_0 / test_loader.dataset.data_path.__len__()
                metric_iou_1 = metric_iou_1 / test_loader.dataset.data_path.__len__()
                ASD = ASD / test_loader.dataset.data_path.__len__()
                HD = HD / test_loader.dataset.data_path.__len__()
                performance = metric_dice_1
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(root_path,
                                                  'AL_iter_{}_dice_{}_{}_{}_{}.pth'.format(
                                                      iter_num, round(best_performance, 4),round(metric_iou_1,4),round(ASD,4),round(HD,4)))
                    save_best = os.path.join(root_path,
                                             'AL_{}_model_{}.pth'.format(AL_time,label_size))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))
                #model.train()
        if iter_num >= max_iterations:
            break
    if iter_num >= max_iterations:
        iterator.close()
    return model,ema_model

def Active_Learning_train_MTCL(train_loader,unlabeled_loader,test_loader,AL_time,
                               max_iterations = 3000,CL_type = 'both',correct_type = 'uncertainty_smooth'):
    weak_supervised_loss = 0.0
    # 构建模型
    params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        classes = 2
    )  # 如果给aux_params，则会输出分类辅助输出
    model_name = 'UNetPlusPlus'
    model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    ema_model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
                                 activation='tanh', aux_params=None)
    for param in ema_model.parameters():  # 教师模型的结构和学生模型一样，但参数要通过指数移动平均更新
        param.detach_()
    model.cuda()
    ema_model = ema_model.cuda()
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_performance = 0.0
    max_iter = max_iterations
    max_epoch = max_iter // len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter(root_path + '/log')
    logging.basicConfig(filename=root_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(train_loader,unlabeled_loader)): #是不是不用Cycle会好些？
            img_batch = torch.cat((sampled_batch[0]['image'].to(torch.float32),sampled_batch[1]['image'].to(torch.float32)),
                                    dim=0)
            unlabeled_volume_batch = sampled_batch[1]['image'].to(torch.float32)
            lab_rect = sampled_batch[0]['label_rect']
            lab = sampled_batch[0]['label']
            if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              lab_batch = lab.cuda().long()
              label_batch = lab_rect.cuda().long()
            label_b = label_batch.shape[0]
            #acc_b = int(label_b/2)  #前面一半的标签数据为高质量
            acc_b = 0
            label_batch = torch.concat([lab_batch[:acc_b],label_batch[acc_b:label_b]],dim=0)
            # 加入噪声
            noise = torch.clamp(torch.randn_like(img_batch) * 0.1, -0.2, 0.2)
            # student model + noise, teacher不加noise
            student_inputs = img_batch + noise
            ema_inputs = img_batch
            # 突然明白为什么输出的维度是【batch_size,num_classes,H,W】了！！！因为它输出的是每个Classes上的概率！我们要取的是更大的那个概率作为分类
            outputs = model(student_inputs)
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
            out = torch.argmax(outputs_soft, dim=1).unsqueeze(1)  # 预测的Label
            #真正的伪标签要用这个Out去更新seg_dict
            with torch.no_grad():
                ema_output = ema_model(ema_inputs.cuda())
            # 损失函数里面有Softmax  监督损失
            loss_seg = F.cross_entropy(outputs[:label_b], label_batch[:label_b])
            loss_seg_dice = losses.dice_loss(outputs_soft[:label_b, 1, :, :], label_batch[:label_b] == 1) + \
                            losses.dice_loss(outputs_soft[:label_b, 0, :, :], label_batch[:label_b] == 0)
            supervised_loss = (loss_seg + loss_seg_dice)

            # 计算不确定度
            T = 10  # 就是根据这10次的结果计算不确定度
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1)
            volume_batch_r_cl = img_batch[:label_b].repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            stride_cl = volume_batch_r_cl.shape[0] // 2
            preds = torch.zeros([stride * T, 2, patch_size[0], patch_size[1]]).cuda()
            preds_cl = torch.zeros([stride_cl * T, 2, patch_size[0], patch_size[1]]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                ema_inputs_cl = volume_batch_r_cl + torch.clamp(torch.randn_like(volume_batch_r_cl) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs.cuda())
                with torch.no_grad():
                    preds_cl[2 * stride_cl * i:2 * stride_cl * (i + 1)] = ema_model(ema_inputs_cl.cuda())
            preds, preds_cl = F.softmax(preds, dim=1), F.softmax(preds_cl, dim=1)
            preds, preds_cl = preds.reshape(T, stride, 2, patch_size[0], patch_size[1]), \
                              preds_cl.reshape(T, stride_cl, 2,patch_size[0], patch_size[1])
            preds, preds_cl = torch.mean(preds, dim=0), torch.mean(preds_cl, dim=0)  # (batch, 2, 112,80)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 112,80)
            uncertainty_cl = -1.0 * torch.sum(preds_cl * torch.log(preds_cl + 1e-6), dim=1, keepdim=True)

            # 扰动一致性损失，ramps up调整一致性损失在整个损失中的权重
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = torch.nn.MSELoss(reduction='mean')(outputs[label_b:], ema_output[label_b:])
            consistency_loss = consistency_weight * consistency_dist

            # L_supervised_loss 置信学习的损失（辅助自去噪置信学习损失）
            noisy_label_batch = label_batch[acc_b:label_b].squeeze(1)  # 这里可以将所有标签数据看成噪声数据损失吗？
            CL_inputs = img_batch[acc_b:label_b].cuda()  # 将标签数据作为Teacher模型的输入

            if iter_num >= max_iterations/3:
                with torch.no_grad():
                    out_main = ema_model(CL_inputs)
                    pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy()

                masks_np = noisy_label_batch.cpu().detach().numpy()

                preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
                preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
                preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
                preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)
                masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)

                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated.shape[0]

                try:
                    if CL_type in ['both']:  #这一步就是置信学习，其实就是找噪声
                        noise = cleanlab.filter.find_label_issues(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                  filter_by='both', n_jobs=1)
                    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                        noise = cleanlab.filter.find_label_issues(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                  filter_by=CL_type, n_jobs=1)

                    confident_maps_np = noise.reshape(label_b-acc_b, patch_size[0], patch_size[1]).astype(np.uint8)
                    # print(confident_maps_np.shape)

                    # label Refinement 对噪声标签进行调整，三种调整方法
                    if correct_type == 'fixed_smooth':
                        smooth_arg = 0.8
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                        print('FS correct the noisy label')
                    elif correct_type == 'uncertainty_smooth':
                        uncertainty_np = uncertainty_cl[acc_b:label_b].cpu().detach().numpy()
                        uncertainty_np_squeeze = np.squeeze(uncertainty_np)
                        smooth_arg = 1 - uncertainty_np_squeeze
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                        # print(corrected_masks_np.shape)
                        print('UDS correct the noisy label')
                    else:
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                        print('Hard correct the noisy label')
                    # 用调整后的标签与Student网络输出构造损失，对应图上的L_{cl}
                    noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft[acc_b:label_b].device.index)
                    loss_ce_weak = Loss_ce(outputs[acc_b:label_b], noisy_label_batch.squeeze(1).long())
                    loss_focal_weak = Loss_focal(outputs[acc_b:label_b], noisy_label_batch.squeeze(1).long())
                    weak_supervised_loss = (loss_ce_weak + loss_focal_weak)
                except Exception as e:
                    print(str(e))
            else:
                weak_supervised_loss = 0.0

            # 总损失
            loss = supervised_loss + consistency_loss + weak_supervised_loss * 5.0

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #这是控制梯度的大小二范数不超过20
            optimizer.step()
            # 更新EMA模型的参数
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            # 学习率衰减
            lr_ = base_lr * (1.0 - iter_num / (max_iter)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num +1
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_entropy: %f,loss_dice: %f,loss_weak:%f' %
                         (iter_num, loss.item(), consistency_dist.item(),
                          loss_seg.item(),
                          loss_seg_dice,weak_supervised_loss))
            print('-' * 50)

            # Validation
            if iter_num > max_iterations/2 and iter_num % 200 == 0 and AL_time == (Max_AL_iter):
                metric_dice_0 = 0.0
                metric_dice_1 = 0.0
                metric_iou_0 = 0.0
                metric_iou_1 = 0.0
                ASD = 0.0
                HD = 0.0
                for index in np.arange(test_loader.dataset.data_path.__len__()):
                    input, label_test,_ = get_data(test_loader.dataset.data_path[index])  #Validation用的是精确标签！
                    input = input.to(torch.float32)  # c, h ,w
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input.unsqueeze(0).cuda())
                        outputs = outputs
                        outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分布
                        out = torch.argmax(outputs_soft, dim=1).squeeze(0).cpu()  # 预测的Label
                        metric_dice_1 += np.sum(metric.binary.dc(out.numpy() == 1, np.array(label_test) == 1))
                        #metric_dice_0 += np.sum(metric.binary.dc(out.numpy() == 0, np.array(label_test) == 0))
                        metric_o, metric_o_1 = compute_IOU(out.numpy(), np.array(label_test))
                        #metric_iou_0 += metric_o
                        metric_iou_1 += metric_o_1

                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                        ASD += avg_surf_dist[1]
                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)  # 95HD
                        HD += hd_dist_95
                #metric_dice_0 = metric_dice_0 / test_loader.dataset.data_path.__len__()
                metric_dice_1 = metric_dice_1 / test_loader.dataset.data_path.__len__()
                #metric_iou_0 = metric_iou_0 / test_loader.dataset.data_path.__len__()
                metric_iou_1 = metric_iou_1 / test_loader.dataset.data_path.__len__()
                ASD = ASD / test_loader.dataset.data_path.__len__()
                HD = HD / test_loader.dataset.data_path.__len__()
                performance = metric_dice_1
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(root_path,
                                                  'AL_iter_{}_dice_{}_{}_{}_{}.pth'.format(
                                                      iter_num, round(best_performance, 4),round(metric_iou_1,4),round(ASD,4),round(HD,4)))
                    save_best = os.path.join(root_path,
                                             'AL_{}_model_{}.pth'.format(AL_time,label_size))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))

                #model.train()
        if iter_num >= max_iterations:
            break
    if iter_num >= max_iterations:
        iterator.close()
    return model,ema_model

def Active_Learning_train_MisMatch(train_loader,unlabeled_loader,test_loader,AL_time,max_iterations = 3000,detach = True):
    model = VNetMisMatchEfficient(n_channels=1, n_classes=num_classes, n_filters=16, normalization='batchnorm',
                                  has_dropout=False)
    model.cuda()
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_performance = 0.0
    max_iter = max_iterations
    max_epoch = max_iter // len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter(root_path + '/log')
    logging.basicConfig(filename=root_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(train_loader,unlabeled_loader)): #是不是不用Cycle会好些？
            img_batch = torch.cat((sampled_batch[0]['image'].to(torch.float32),sampled_batch[1]['image'].to(torch.float32)),
                                    dim=0)
            unlabeled_volume_batch = sampled_batch[1]['image'].to(torch.float32)
            lab = sampled_batch[0]['label_rect']
            if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              label_batch = lab.cuda().long()
            # 突然明白为什么输出的维度是【batch_size,num_classes,H,W】了！！！因为它输出的是每个Classes上的概率！我们要取的是更大的那个概率作为分类
            outputs_p, outputs_n = model(img_batch)
            outputs_soft_p = F.softmax(outputs_p, dim=1)
            outputs_soft_n = F.softmax(outputs_n, dim=1)
            outputs_soft_avg = (outputs_soft_p + outputs_soft_n) / 2
            out = torch.argmax(outputs_soft_avg, dim=1).unsqueeze(1)  # 预测的Label
            label_b = label_batch.shape[0]
            # 损失函数里面有Softmax  监督损失
            loss_seg = 0.5 * F.cross_entropy(outputs_p[:label_b], label_batch[:label_b]) + 0.5 * F.cross_entropy(
                outputs_n[:label_b], label_batch[:label_b])
            loss_seg_dice = losses.dice_loss(outputs_soft_avg[:label_b, 1, :, :], label_batch[:label_b] == 1) + \
                            losses.dice_loss(outputs_soft_avg[:label_b, 0, :, :], label_batch[:label_b] == 0)
            supervised_loss = (loss_seg + loss_seg_dice)

            # 扰动一致性损失，ramps up调整一致性损失在整个损失中的权重
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            outputs_p_u = outputs_p[label_b:]
            outputs_n_u = outputs_n[label_b:]

            if detach is True:
                consistency_dist = 0.5 * torch.nn.MSELoss(reduction='mean')(outputs_p_u, outputs_n_u.detach()) + \
                                   0.5 * torch.nn.MSELoss(reduction='mean')(outputs_n_u, outputs_p_u.detach())
            else:
                consistency_dist = 0.5 * torch.nn.MSELoss(reduction='mean')(outputs_p_u, outputs_n_u) + \
                                   0.5 * torch.nn.MSELoss(reduction='mean')(outputs_n_u, outputs_p_u)
            consistency_loss = consistency_weight * consistency_dist

            loss = supervised_loss + consistency_dist

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #这是控制梯度的大小二范数不超过20
            optimizer.step()
            # 更新EMA模型的参数

            # 学习率衰减
            lr_ = base_lr * (1.0 - iter_num / (max_iter)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num +1
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_entropy: %f,loss_dice: %f' %
                         (iter_num, loss.item(), consistency_dist.item(),
                          loss_seg.item(),
                          loss_seg_dice.item()))

            #Loss.append([loss.item(),supervised_loss.item(),loss_seg_dice.item(),consistency_loss.item()])

            print('-' * 50)

            # Validation
            if iter_num > max_iterations/2 and iter_num % 300 == 0 and AL_time == (Max_AL_iter):
                metric_dice_0 = 0.0
                metric_dice_1 = 0.0
                metric_iou_0 = 0.0
                metric_iou_1 = 0.0
                ASD = 0.0
                HD = 0.0
                for index in np.arange(test_loader.dataset.data_path.__len__()):
                    input, label_test,_ = get_data(test_loader.dataset.data_path[index])  #Validation用的是精确标签！
                    input = input.to(torch.float32)  # c, h ,w
                    model.eval()
                    with torch.no_grad():
                        outputs_p, outputs_n = model(input.unsqueeze(1).cuda())
                        outputs_soft_p = F.softmax(outputs_p, dim=1)
                        outputs_soft_n = F.softmax(outputs_n, dim=1)
                        outputs_soft = (outputs_soft_p + outputs_soft_n) / 2
                        out = torch.argmax(outputs_soft, dim=1).squeeze(0).squeeze(0).cpu()
                        metric_dice_1 += np.sum(metric.binary.dc(out.numpy() == 1, np.array(label_test) == 1))
                        #metric_dice_0 += np.sum(metric.binary.dc(out.numpy() == 0, np.array(label_test) == 0))
                        metric_o, metric_o_1 = compute_IOU(out.numpy(), np.array(label_test))
                        #metric_iou_0 += metric_o
                        metric_iou_1 += metric_o_1

                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                        ASD += avg_surf_dist[1]
                        surface_distances = surfdist.compute_surface_distances(
                          label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                          spacing_mm=(1.0, 1.0))
                        hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)  # 95HD
                        HD += hd_dist_95
                #metric_dice_0 = metric_dice_0 / test_loader.dataset.data_path.__len__()
                metric_dice_1 = metric_dice_1 / test_loader.dataset.data_path.__len__()
                #metric_iou_0 = metric_iou_0 / test_loader.dataset.data_path.__len__()
                metric_iou_1 = metric_iou_1 / test_loader.dataset.data_path.__len__()
                ASD = ASD / test_loader.dataset.data_path.__len__()
                HD = HD / test_loader.dataset.data_path.__len__()
                performance = metric_dice_1
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(root_path,
                                                  'AL_iter_{}_dice_{}_{}_{}_{}.pth'.format(
                                                      iter_num, round(best_performance, 4),round(metric_iou_1,4),round(ASD,4),round(HD,4)))
                    save_best = os.path.join(root_path,
                                             'AL_{}_model_{}.pth'.format(AL_time,label_size))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))
    if iter_num >= max_iterations:
      iterator.close()
    return model, model

#x = np.arange(IoU_npl_dict[0].__len__())
#plt.plot(x, IoU_npl_dict[0], 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='一些数字')


if __name__ == '__main__':
   Time = []
   train_dataset, test_dataset, unlabeled_dataset = DataLoad(root_path,True)
   #train_dataset = FullDataset(root_path, True, type='Full_Train')  #全监督学习
   test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, num_workers=0,
                            pin_memory=False)
   #model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
   #                         decoder_use_batchnorm=True,
   #                         decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
   #                         activation='tanh', aux_params=None)
   #ema_model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_depth=4, encoder_weights='imagenet',
   #                             decoder_use_batchnorm=True,
   #                             decoder_channels=[128, 64, 32, 16], in_channels=1, classes=num_classes,
   #                             activation='tanh', aux_params=None)
   #for param in ema_model.parameters():  # 教师模型的结构和学生模型一样，但参数要通过指数移动平均更新
   #    param.detach_()
   #ema_model = ema_model.cuda()
   for AL_time in np.arange(Max_AL_iter+1):
     train_loader = DataLoader(dataset=train_dataset, batch_size=labeled_bs, num_workers=0,
                          pin_memory=False)
     unlabeled_loader = DataLoader(dataset=unlabeled_dataset,batch_size=batch_size-labeled_bs,num_workers=0,
                                pin_memory=False)

     Net = 'Not_MisMatch'
     start_time = time.time()
     model,ema_model = Active_Learning_train(train_loader,unlabeled_loader,test_loader,AL_time,True)
     #model, ema_model = Active_Learning_train_MeanTeacher(train_loader, unlabeled_loader, test_loader, AL_time,max_iterations=1500)
     #model, ema_model = Active_Learning_train_MTCL(train_loader, unlabeled_loader, test_loader, AL_time,max_iterations=1500)
     #model,ema_model = Active_Learning_train_MisMatch(train_loader,unlabeled_loader,test_loader,AL_time,max_iterations = 1500,detach = True)
     #Net = 'MisMatch'
     end_time = time.time()
     Time.append(round(end_time-start_time,2))
     Unlabeled_list = unlabeled_loader.dataset.data_path
     if AL_time != (Max_AL_iter):
      quary_method = 'KL'
      if Net == 'MisMatch':
          chosen = query_stg.Query(model,model,unlabeled_dataset, train_dataset, label_size, 'KL_MisMatch',False)
      else:
          chosen = query_stg.Query(model,ema_model,unlabeled_dataset,train_dataset,label_size,quary_method,dropout = False)
      del model
      del ema_model
      train_dataset.update_dict(chosen, Unlabeled_list)
      unlabeled_dataset.delete_dict(chosen)
      labeled_bs += 2
   print(Time)