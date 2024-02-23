import cv2
#import us
import numpy as np

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

#该函数用于对医院原始jpg图像进行脱敏处理
#输入三通道原始图像'image'
#输出单通道脱敏后灰度图像'warped_image'
def desensitize(image):
    # STEP 1: 强化轮廓对比度
    # Read the image and detect edges
    image_copy = image.copy()#三通道原图拷贝
    # image = cv2.resize(image, (1500, 800))
    image1=image.copy()#单通道原图拷贝
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #开始处理
    im=cv2.adaptiveThreshold(image_gray,255,0,cv2.THRESH_BINARY,7,0)
    im = 255 - im
    image_blurred = cv2.GaussianBlur(im, (11, 11), 0)
    image_edge = cv2.Canny(image_blurred, 25, 200)
    ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # 大小为5，形状椭圆
    im_morphology = cv2.morphologyEx(image_edge, cv2.MORPH_CLOSE, ele)
    ele1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # 大小为5，形状椭圆
    im_morphology = cv2.morphologyEx(im_morphology, cv2.MORPH_OPEN, ele1)
    # STEP 2: 提取轮廓，按面积排序
    cnts = cv2.findContours(im_morphology.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:15]
    # Step 3: 识别轮廓的外接多边形，提取四边形的一组跳出
    k = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)#计算轮廓的周长，第一个参数表示轮廓，第二个参数表示是否轮廓封闭，True表示计算时认为轮廓首位相连，false反之
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)#以指定精度近似生成顶点最少的多变形曲线，第二个参数为曲线与直线相差的最大距离
        if len(approx) == 4:
            screenCant = approx
            cv2.drawContours(image1, [screenCant], -1, (0, 255, 0), 2)
            k = 1
            break
    # Step 4: Get the perspective transformation
    if k==1:
      warped_image = four_point_transform(image, screenCant.reshape(4,2))
    else:warped_image = image_copy
    return warped_image


#image_PATH = "4.JPG"
#image = cv2.imread(image_PATH)
#a=desensitize(image)
#print(a.shape)
#cv2.imshow('intput',image)
#cv2.imshow('output',a)
#cv2.waitKey(0)