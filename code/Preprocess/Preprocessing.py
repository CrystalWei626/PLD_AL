import os
import cv2
import sys
sys.path.append('data/tyc/code/Preprocess')
import desensitize
import Interpolate
import numpy as np
import SimpleITK as sitk


def maxtrix_min(image):
    h, w = image.shape
    print(h, w)
    min = image[1][1]
    i = 0
    j = 0
    jmin = 0
    imin = 0
    for j in range(0, w):
        for i in range(0, h):
            print(i, j)
            if image[i][j] < min:
                min = image[i][j]
                imin = i
                jmin = j
    return (jmin, imin)

def takeSecond(elem):
    return elem[1]
def find_artery_area(im2):
    a, b, c, d = cv2.connectedComponentsWithStats(im2,
                                                  connectivity=4)
    print(d)  # 质心
    h, w = b.shape
    fill_artery = np.ones(b.shape, dtype=np.uint8)
    centroid_list = d.tolist()
    centroid_list2 = centroid_list.copy()
    centroid_list.pop(0)  # 去掉背景连通域
    centroid_list.sort(key=takeSecond, reverse=False)  #
    artery_val = centroid_list2[1]
    artery_index = centroid_list2.index(artery_val)
    area = c[artery_index][4]
    for i in range(0, w):
      for j in range(0, h):
        if b[j][i] == artery_index or b[j][i] == 0:
          fill_artery[j, i] = 255
        else:
          fill_artery[j][i] = 0
    print(artery_index)
    print(area)
    return fill_artery

#将dcm文件转换成Img
if __name__ == '__main__':
  base_dir = "data/tyc/data_InHouse/swct1"
  data_list = os.listdir(base_dir)
  for i in data_list:
     dcm_PATH = os.path.join(base_dir, i)
     data = sitk.ReadImage(dcm_PATH)
     data_arr = sitk.GetArrayFromImage(data)
     for z in np.arange(data_arr.shape[0]):
       h,w = data_arr.shape[1],data_arr.shape[2]
       if len(data_arr.shape)>3:
           data_arr_bin = cv2.cvtColor(data_arr[z], cv2.COLOR_RGB2GRAY)
       else: data_arr_bin = data_arr[z]
       data_a = np.array(data_arr_bin).reshape(h,w)
       cv2.imwrite('data/tyc/data_InHouse/swct_img\\{}.png'.format(i.split('.')[0]), data_a)

  #数据脱敏
  base_dir = "data/tyc/data_InHouse/swct_img"
  data_list = os.listdir(base_dir)
  for i in data_list:
      img_PATH = os.path.join(base_dir, i)
      image = cv2.imread(img_PATH)
      a = desensitize.desensitize(image)
      cv2.imwrite('data/tyc/data_InHouse/swct_img_des\\{}.png'.format(i.split('.')[0]), a)

#第一步：超声原始图像预处理a
  base_dir = "data/tyc/data_InHouse/swct_ROI/N"
  data_list = os.listdir(base_dir)
#五个模板
  for t in np.arange(5):
# 第二部：导入匹配模板b
   bb = cv2.imread("data/tyc/data_InHouse/swct_ROI/Temp{}.png".format(t))
   b = cv2.cvtColor(bb, cv2.COLOR_RGB2GRAY)
   hh, ww = b.shape
   for i in data_list:
     image_PATH = os.path.join(base_dir,i)
     image = cv2.imread(image_PATH)
     a = image
     a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
     a_color=np.copy(a)
     a_copy2 = np.copy(a)
     h, w = a.shape

  #第三步：模板匹配  匹配结果：result1  最优点：minval1  提示框展示：rectangle  截取：ROI
     print(i)
     result1=np.zeros((h-hh+1,w-ww+1),dtype=np.float32)#result结果图需新建（匹配高-模板高+1，匹配款-模板宽+1）32位浮点图
     cv2.matchTemplate(a,b,method=cv2.TM_SQDIFF,result=result1)
     minval1=maxtrix_min(result1)
     rectangle=cv2.rectangle(a,(minval1[0],minval1[1]),(minval1[0]+ww,minval1[1]+hh),(255,255,255),2)#在结果图中找到最匹配的部分框出来
     ROI=a_color[minval1[1]:minval1[1]+hh,minval1[0]:minval1[0]+ww]#提取出这一部分匹配结果，作为单张图片
     cv2.imwrite('data/tyc/data_InHouse/swct_ROI/{}/{}_rect.png'.format(t,i.split('.')[0]), rectangle)
     cv2.imwrite('data/tyc/data_InHouse/swct_ROI/{}/{}.png'.format(t,i.split('.')[0]), ROI)

  dir_list = "data/tyc/data_InHouse/ROI"
  for i in os.listdir(dir_list):
    img_dir = os.path.join(dir_list,i)
    img = cv2.imread(img_dir)
    a_LI = []
    b_LI = []
    a_MA = []
    b_MA = []
    f1 = open("{}-LI.txt", "w")
    for x,y in zip(a_LI,b_LI):
      f1.write(x+'\t'+y+'\n')
    f1.close()

    f2 = open("{}-MA.txt", "w")
    for x,y in zip(a_MA,b_MA):
      f2.write(x+'\t'+y+'\n')
    f2.close()

  ROI_dir = "data/tyc/data_InHouse/ROI"
  segment_dir = "data/tyc/data_InHouse/segment"
  label_dir = "data/tyc/data_InHouse/labels"
  for ROI_d in os.listdir(ROI_dir):
      ROI = os.path.join(ROI_dir, ROI_d)
      ROI_img = cv2.imread(ROI)
      txt_LI = os.path.join(segment_dir, ROI_d.split('.')[0] + '_LI.txt')
      txt_MA = os.path.join(segment_dir, ROI_d.split('.')[0] + '_MA.txt')
      h, w = ROI_img.shape[0], ROI_img.shape[1]
      l = 20
      Interpolate.Get_Label(h, w, txt_LI, txt_MA, label_dir, l)
