import numpy as np
import torch.nn.functional as F
import torch
import pdb
from scipy import stats
import sys
sys.path.append('data/tyc/code/Dataset')
from Dataset.Get_data import get_data
import torch.nn as nn

def pairwise_distances(a, b): #计算距离 a：n1 * c * h *w  b：n2*c*h*w  #计算特征图之间的像素级欧氏距�?
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    dist = np.zeros((a.size(0), b.size(0)), dtype=np.float) #dist：n1*n2
    for i in range(b.size(0)): #�?0�?n2-1
        b_i = b[i]  #bi : c *h *w
        kl1 = a * torch.log(a / b_i) #其实就是两个相反的KL距离平均
        kl2 = b_i * torch.log(b_i / a)
        dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) +\
                     0.5 * (torch.sum(kl2, dim=1))
    return dist

def select_coreset(X, X_set, n):
      m = np.shape(X)[0] #m表示未标记数据的数量
      if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
      else:
            dist_ctr = pairwise_distances(X, X_set) #算标记预测和未标记预测的距离
            min_dist = np.amin(dist_ctr, axis=1) #得到最小的距离

      idxs = []

      print('selecting coreset...')
      for i in range(n):
            idx = min_dist.argmax() #在一群最小距离中找最大的�?
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :]) #更新距离�?
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0]) #更新距离

      return idxs

#太慢！换成用Embedding
def CDAL(model,unlabeled_dataset,train_dataset,n): #n表示选择的数�?
       chosen = []
       embeddings_label = np.zeros((train_dataset.data_path.__len__(),128))
       for index in np.arange(train_dataset.data_path.__len__()):
             input,label,_ = get_data(train_dataset.data_path[index])
             input = input.to(torch.float32) #c, h ,w
             model.eval()
             with torch.no_grad():
                   outputs = model(input.unsqueeze(0).cuda())
                   outputs_soft = torch.softmax(outputs, dim=1)
                   embd = model.encoder(input.unsqueeze(0).cuda())[4]
                   embd = embd.cpu().detach().numpy()
                   embd = np.sum(embd, axis=1)
                   embeddings_label[index, :] = embd.reshape(32 * 4)

       embeddings_unlabel = np.zeros((unlabeled_dataset.data_path.__len__(), 128))
       min_dis = torch.zeros(unlabeled_dataset.data_path.__len__(),)
       for index in np.arange(unlabeled_dataset.data_path.__len__()):
             input,label,_ = get_data(unlabeled_dataset.data_path[index])
             input = input.to(torch.float32) #c, h ,w
             model.eval()
             with torch.no_grad():
                   outputs = model(input.unsqueeze(0).cuda())
                   outputs_soft = torch.softmax(outputs, dim=1)
                   embd = model.encoder(input.unsqueeze(0).cuda())[4]
                   embd = embd.cpu().detach().numpy()
                   embd = np.sum(embd, axis=1)
                   embeddings_unlabel[index, :] = embd.reshape(32 * 4)
                   dist_matrix = torch.pairwise_distance(torch.from_numpy(embd.reshape(1,32 * 4)), torch.from_numpy(embeddings_label))
                   dis_min = min(dist_matrix)
                   min_dis[index] = dis_min

       m = min_dis.__len__()
       for i in range(n):
           idx = min_dis.argmax()  # 在一群最小距离中找最大的�?
           chosen.append(idx)
           dist_new_ctr = torch.pairwise_distance(torch.from_numpy(embeddings_unlabel),
                                                  torch.from_numpy(embeddings_unlabel[[idx], :]))  # 更新距离�?
           for j in range(m):
               min_dis[j] = min(min_dis[j], dist_new_ctr[j])  # 更新距离

       chosen = [c.detach().numpy()  for c in chosen]
       return chosen

#Badge：对特征进行KMeans聚类
def init_centers(X, K):  #这里的Embedding是梯度嵌入（也可以用普通嵌入吧），这就是个KMeans的函�?
    ind = np.argmax([np.linalg.norm(s, 2) for s in X]) #Embedding中最大的先选了再说  先求二范数，选最大的二范�?
    mu = [X[ind]] #mu用于储存所有备选的Embedding
    indsAll = [ind] #用于储存下标
    centInds = [0.] * len(X) #用于保存不同的中�?
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K: #K是最大的选择数量，不能超过它
        if len(mu) == 1:
            D2 = pairwise_distances(X, np.array(mu)).ravel().astype(float) #用于计算X到mu的距�?
        else:
            newD = pairwise_distances(X, np.array([mu[-1]])).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

def Badge(model,unlabeled_dataset,train_dataset,n):
    embeddings = np.zeros((unlabeled_dataset.data_path.__len__(),128))
    for index in np.arange(unlabeled_dataset.data_path.__len__()):
        input, label, _ = get_data(unlabeled_dataset.data_path[index])
        nLab = len(np.unique(label))
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        model.encoder.requires_grad_(False)
        model.zero_grad()
        x_in = input.unsqueeze(0).cuda()
        outputs = model(x_in)
        y = model.forward(x_in)
        outputs.backward(y)

        embd = model.segmentation_head[0].weight.grad
        embd = embd.cpu().detach().numpy()
        embd = np.sum(embd, axis=1)
    chosen = init_centers(embeddings, n)
    return chosen

#CoreSet
def furthest_first(X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)
        idxs = []
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        return idxs

def CoreSet(model,unlabeled_dataset,train_dataset,n):
    embeddings_label = np.zeros((train_dataset.data_path.__len__(), 128))
    for index in np.arange(train_dataset.data_path.__len__()):
        input, label, _ = get_data(train_dataset.data_path[index])
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            embd = model.encoder(input.unsqueeze(0).cuda())[4]
            embd = embd.cpu().detach().numpy()
            embd = np.sum(embd, axis=1)
            embeddings_label[index, :] = embd.reshape(32 * 4)

    embeddings_unlabel = np.zeros((unlabeled_dataset.data_path.__len__(), 128))
    min_dis = torch.zeros(unlabeled_dataset.data_path.__len__(), )
    for index in np.arange(unlabeled_dataset.data_path.__len__()):
        input, label, _ = get_data(unlabeled_dataset.data_path[index])
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            outputs_soft = torch.softmax(outputs, dim=1)
            embd = model.encoder(input.unsqueeze(0).cuda())[4]
            embd = embd.cpu().detach().numpy()
            embd = np.sum(embd, axis=1)
            embeddings_unlabel[index, :] = embd.reshape(32 * 4)
            dist_matrix = torch.pairwise_distance(torch.from_numpy(embd.reshape(1, 32 * 4)),
                                                  torch.from_numpy(embeddings_label))
            dis_min = min(dist_matrix)
            min_dis[index] = dis_min

    chosen = furthest_first(embeddings_unlabel, embeddings_label, n)
    return chosen

#Random
def find_un(n, un):
    chosen = []
    for _ in range(n):
        number = min(un[un > 0])
        index = un.tolist().index(number)
        un[index] = 10000000
        chosen.append(index)
    return chosen

def Random(model,unlabeled_dataset,train_dataset,n):
    import random
    un = np.array([random.randint(0, 1000) for i in range(unlabeled_dataset.data_path.__len__())])
    chosen = find_un(n, un)
    return chosen

def UnCertainty(model,unlabeled_dataset,train_dataset,n):
    idxs_unlabeled = np.arange(unlabeled_dataset.data_path.__len__())
    for index in np.arange(unlabeled_dataset.data_path.__len__()):
        input, label, _ = get_data(unlabeled_dataset.data_path[index])
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        # ema_model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            #outputs = outputs[0]
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分�?
            if index == 0:
                Uncertainty_ALL = torch.sum(
                    torch.sum(torch.sum((outputs_soft - 0.5) ** 2, dim=1), dim=1), dim=1)
            else:
                un = torch.sum(torch.sum(torch.sum((outputs_soft - 0.5) ** 2, dim=1), dim=1), dim=1)
                Uncertainty_ALL = torch.cat((Uncertainty_ALL, un), dim=0)

    un = Uncertainty_ALL.cpu()
    selected = un.sort(descending = False)[1][:n]
    chosen = idxs_unlabeled[selected]
    return chosen

#�?
def Entropy(model,unlabeled_dataset,train_dataset,patch_size,n):
    idxs_unlabeled = np.arange(unlabeled_dataset.data_path.__len__())
    probs_u = torch.zeros(unlabeled_dataset.data_path.__len__(), 2, patch_size[0], patch_size[1])
    for index in np.arange(unlabeled_dataset.data_path.__len__()):
        input, label, _ = get_data(unlabeled_dataset.data_path[index])
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            outputs_soft = torch.softmax(outputs, dim=1)
            probs_u[index,:,:,:] = outputs_soft

    log_probs = torch.log(probs_u)
    U = (probs_u * log_probs).sum(1).sum(1).sum(1)
    selected = U.sort(descending = True)[1][:n]
    chosen = idxs_unlabeled[selected]
    return chosen

#KL散度
def KL(model,ema_model,unlabeled_dataset,train_dataset,patch_size,n,dropout = False):
    idxs_unlabeled = np.arange(unlabeled_dataset.data_path.__len__())
    probs_u = torch.zeros(unlabeled_dataset.data_path.__len__(),2,patch_size[0],patch_size[1])
    probs_u_ema = torch.zeros(unlabeled_dataset.data_path.__len__(),2,patch_size[0],patch_size[1])
    min_dis = torch.zeros(unlabeled_dataset.data_path.__len__(), )
    for index in np.arange(unlabeled_dataset.data_path.__len__()):
        input, label, _ = get_data(unlabeled_dataset.data_path[index])
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        ema_model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            if dropout == True:
                outputs = outputs[0]
            outputs_soft = torch.softmax(outputs, dim=1)  # 预测的概率分�?
            probs_u[index,:,:,:] = outputs_soft
            outputs_ema = ema_model(input.unsqueeze(0).cuda())
            if dropout == True:
                outputs_ema = outputs_ema[0]
            outputs_soft_ema = torch.softmax(outputs_ema, dim=1)
            probs_u_ema[index,:,:,:] = outputs_soft_ema

    sm = nn.Softmax(dim=1)
    log_sm = nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')
    # kl 近似等于 variance
    kl = torch.sum(kl_distance(log_sm(probs_u.cuda()), sm(probs_u_ema.cuda())), dim=1)  # 前面的是student model, 被指�?
    Variance = torch.sum(torch.sum(kl,dim=1),dim=1)
    selected = Variance.sort(descending = True)[1][:n]  #降序排列，应该是方差越大的越不确定吧
    selected = selected.cpu()
    chosen = idxs_unlabeled[selected]
    return chosen


def Query(model,ema_model,unlabeled_dataset,train_dataset,patch_size,label_size,quary_method,dropout):
    if quary_method == 'Random':
        chosen = Random(model,unlabeled_dataset,train_dataset,label_size)
    if quary_method == 'UnCertainty':
        chosen = UnCertainty(model,unlabeled_dataset,train_dataset,label_size)
    if quary_method == 'KL':
        chosen = KL(model,ema_model,unlabeled_dataset,train_dataset,patch_size,label_size,dropout)
    if quary_method == 'Entropy':
        chosen = Entropy(model,unlabeled_dataset,train_dataset,patch_size,label_size)
    if quary_method == 'CoreSet':
        chosen = CoreSet(model,unlabeled_dataset,train_dataset,label_size)
    if quary_method == 'CDAL':
        chosen = CDAL(model,unlabeled_dataset,train_dataset,label_size)
    if quary_method == 'Badge':
        chosen = Badge(model,unlabeled_dataset,train_dataset,label_size)
    return chosen


