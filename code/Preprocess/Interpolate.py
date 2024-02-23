import os
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import cv2

def read_txt(txt_dir):
  with open(txt_dir, 'r', encoding='utf-8') as f:
    data = f.readlines()
  X = []
  Y = []
  for d in data:
    # print(d)
    X.append(int(d.split(',')[0]))
    Y.append(int(d.split(',')[1]))
  return X,Y

def Search_Y(inter,x,w):
  D = 1000
  x_y = x
  for x_s in np.arange(x-50,x+50):
    if x_s>=0 and x_s<w and inter[x_s]>0:
      d = np.abs(x-x_s)
      if d<D:
        x_y = x_s
        D = d
  return x_y


def get_line(x1, y1, x2, y2):
  points = []
  issteep = abs(y2 - y1) > abs(x2 - x1)
  if issteep:
    x1, y1 = y1, x1
    x2, y2 = y2, x2
  rev = False
  if x1 > x2:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
    rev = True
  deltax = x2 - x1
  deltay = abs(y2 - y1)
  error = int(deltax / 2)
  y = y1
  ystep = None
  if y1 < y2:
    ystep = 1
  else:
    ystep = -1
  for x in range(x1, x2 + 1):
    if issteep:
      points.append((y, x))
    else:
      points.append((x, y))
    error -= deltay
    if error < 0:
      y += ystep
      error += deltax
  # Reverse the list if the coordinates were reversed
  if rev:
    points.reverse()
  return points


def Get_Label(h,w,segment_dir_LI,segment_dir_MA,label_dir,l): #l表示那个不连接的阈值
  label = np.zeros((h,w))
  #t_dir_1 = os.path.join(segment_dir,segment_dir_LI)
  #t_dir_2 = os.path.join(segment_dir,segment_dir_MA)
  X_LI,Y_LI = read_txt(segment_dir_LI)
  X_MA,Y_MA = read_txt(segment_dir_MA)
  li = np.full(w, np.nan)
  ma = np.full(w, np.nan)
    #排序
  S_LI = sorted(enumerate(X_LI), key=lambda X_LI: X_LI[1])
  X_LI = [X_LI[s[0]] for s in S_LI]
  Y_LI = [Y_LI[s[0]] for s in S_LI]
  S_MA = sorted(enumerate(X_MA), key=lambda X_MA: X_MA[1])
  X_MA = [X_MA[s[0]] for s in S_MA]
  Y_MA = [Y_MA[s[0]] for s in S_MA]
  for (x1,y1) in zip(X_LI,Y_LI):
      li[x1] = y1
  for (x1,y1) in zip(X_MA,Y_MA):
      ma[x1] = y1
  li = pd.Series(li)
  li_inter = li.interpolate(method='akima', limit=l)
  ma = pd.Series(ma)
  ma_inter = ma.interpolate(method='akima', limit=l)
  for x in np.arange(w):
        if li_inter[x]>0 and ma_inter[x]>0:
          for y in np.arange(int(li_inter[x]),int(ma_inter[x])+1):
            label[y,x] = 255
        if li_inter[x]>0 and np.isnan(ma_inter[x]):
          x_y = Search_Y(ma_inter,x,w)
          point_list = get_line(int(x),int(li_inter[x]),int(x_y),int(ma_inter[x_y]))
          for p in point_list:
            label[p[1],p[0]] = 255
        if np.isnan(li_inter[x]) and ma_inter[x]>0:
          x_y = Search_Y(li_inter, x,w)
          point_list = get_line(int(x_y),int(li_inter[x_y]),int(x),int(ma_inter[x]))
          for p in point_list:
            label[p[1],p[0]] = 255
  label = np.nan_to_num(label)
  #cv2.imwrite(label_dir+'/t.png', label)
  Image.fromarray(label).convert('RGB').save(label_dir+'/t.png')

def write_in_h5(h5_save_path, images, labels, label_rect):
    with h5py.File(h5_save_path, mode='w') as hf:
      hf.create_dataset(name="image", dtype=float, data=images)
      hf.create_dataset(name="label", dtype=float, data=np.array(labels).astype(bool).astype(int))
      hf.create_dataset(name="label_rect", dtype=float, data=np.array(label_rect).astype(bool).astype(int))
      #hf.create_dataset(name="label_bis", dtype=float, data=np.array(label_bis).astype(bool).astype(int))
    print(h5_save_path)

def Get_h5files(h5_file_dir,label_dir,label_dir_rect,x):
  img_dir  = label_dir+'/img.png'
  lab_dir = label_dir+'/t.png'
  lab_dir_rect = label_dir_rect+'/t.png'
  h5_save_path = h5_file_dir+'/'+x+'.h5'
  images = cv2.imread(img_dir)
  labels = cv2.imread(lab_dir)
  label_rect = cv2.imread(lab_dir_rect)
  write_in_h5(h5_save_path,images,labels,label_rect)


label_dir_list = r'D:\BaiduNetdiskDownload\swct_ROI\dataset_labeled' #标注1
label_dir_list_rect = r'D:\BaiduNetdiskDownload\swct_ROI\dataset_labeled_rect'
h5_file_dir = r"D:\BaiduNetdiskDownload\swct_ROI\h5"
for x in os.listdir(label_dir_list):
  label_dir = os.path.join(label_dir_list,x)
  segment_dir_LI = label_dir+'/label_point1.txt'
  segment_dir_MA = label_dir+'/label_point2.txt'
  h, w = 80,128
  l = 20
  Get_Label(h, w, segment_dir_LI, segment_dir_MA, label_dir, l)
  label_dir_rect = os.path.join(label_dir_list_rect, x)
  segment_dir_LI_rect = label_dir_rect + '/label_point1.txt'
  segment_dir_MA_rect = label_dir_rect + '/label_point2.txt'
  Get_Label(h, w, segment_dir_LI_rect, segment_dir_MA_rect, label_dir_rect, l)
  Get_h5files(h5_file_dir, label_dir, label_dir_rect, x)


