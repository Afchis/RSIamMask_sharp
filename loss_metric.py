import torch
import torch.nn.functional as F

from args import *


########## Mask losses: ##########
def l2_loss(x, y, d, smooth = 1.):
    ones = (d*torch.ones_like(x)).sum()
    out = (d*(x - y)**2).sum()
    out = (out + smooth)/(ones + smooth)
    return out

def bce_loss(x, y, d):
    bce_loss =  F.binary_cross_entropy(x, y).mean()
    return bce_loss

def soft_margin_loss(x, y, d):
    x = torch.nn.functional.log_softmax(x, dim=1)
    soft_loss = torch.nn.MultiLabelSoftMarginLoss()
    return soft_loss(x, y)

def cross_entropy_loss(x, y, d):
    cross_entropy = torch.nn.CrossEntropyLoss()
    return cross_entropy(x, y)

def dice_loss(x, y, d, smooth = 1.):
    
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    #print(dice_loss.mean().item())
    return dice_loss.mean()

def dice_combo_loss(x, y, d, bce_weight=0.5):
    dice_combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * dice_loss(x, y, d)
    return dice_combo_loss

def iou_loss(x, y, d, threshold=0.5, smooth = 1.):
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    union = x_sum + y_sum - intersection
    iou_metric = 1 - ((intersection + smooth) / (union + smooth))
    #print(dice_loss.mean().item())
    return iou_metric.mean()

def iou_combo_loss(x, y, d, bce_weight=0.2):
    iou_combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * iou_loss(x, y, d)
    return iou_combo_loss

def l2_combo_loss(x, y, d, bce_weight=0.5):
    l2_combo_loss = bce_weight * l2_loss(x, y, d) + bce_weight * bce_loss(x, y, d)# + dice_loss(x, y, d)
    # print(l2_combo_loss)
    return l2_combo_loss


########## Score losses: ##########
def score_loss(xx, yy):
    bce_loss =  torch.nn.BCELoss()
    return  bce_loss(xx, yy)


########## All losses: ##########
def all_losses(x, y, d, xx, yy):
    # print(x.shape, y.shape, d.shape, xx.shape, yy.shape)
    x = x.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    y = y.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    d = d.reshape(-1, 1, TARGET_SIZE, TARGET_SIZE)
    xx, yy = xx.reshape(-1, NUM_CLASSES, 17, 17), yy.reshape(-1, NUM_CLASSES, 17, 17)
     ##### LOSS #####
      ##### LOSS #####
       ##### LOSS #####
    all_losses =  bce_loss(x, y, d) + score_loss(xx, yy) ##### LOSS #####
        ##### LOSS #####
       ##### LOSS #####
      ##### LOSS #####
    return  all_losses


########## IOU metric: ##########
def iou_metric(x, y, d, threshold=0.5, smooth = 1.):
    x = x.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    y = y.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    # x = x[:, 1:]
    # y = y[:, 1:]
    x = (x > threshold).float()
    
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    union = x_sum + y_sum - intersection
    iou_metric = ((intersection + smooth) / (union + smooth))
    #print(dice_loss.mean().item())
    return iou_metric.mean()

if __name__ == '__main__':
    x = torch.rand([1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE])
    y = torch.rand([1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE])
    d = torch.rand([1, 1, TARGET_SIZE, TARGET_SIZE])
    print(iou_metric(x, y, d))
