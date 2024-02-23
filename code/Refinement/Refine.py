from scipy.optimize import curve_fit
from pylab import *


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
    return IoU[0],IoU[1]


def curve_func(x, a, b, c):
    return a * (1 - np.exp(-1 / c * x ** b))

def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(y)),
                           absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]),maxfev = 10000)
    return tuple(popt)

def derivation(x, a, b, c):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))

def label_update_epoch(ydata_fit,  num_iter_per_epoch,current_epoch,threshold, eval_interval=100):
    xdata_fit = np.arange(1, current_epoch+2)
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    epoch = np.arange(1, current_epoch+2)
    abs_change = abs(abs(derivation(epoch, a, b, c))[current_epoch] - abs(derivation(epoch, a, b, c))[current_epoch-1])
    print(abs_change)
    if (abs_change <= threshold):
        update_epoch = True
    else:
        update_epoch = False
    return update_epoch,abs_change

def if_update(iou_value, current_epoch, num_iter_per_epoch,threshold):
    update_epoch,abs_change = label_update_epoch(iou_value,num_iter_per_epoch,current_epoch,threshold=threshold)
    return update_epoch,abs_change
