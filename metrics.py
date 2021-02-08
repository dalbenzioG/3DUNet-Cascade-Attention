import torch
import numpy as np
from torch.nn import functional as F

epsilon = 1.0e-6


def multi_acc(pred, label):
    probs = torch.log_softmax(pred, dim=1)
    _, tags = torch.max(probs, dim=1)
    corrects = torch.eq(tags, label).int()
    acc = corrects.sum() // corrects.numel()
    return acc


def dice_coefficient(y_true, y_pred, num_classes=3, axis=(0, 2, 3, 4),
                     epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.
    """
    y_pred = F.softmax(y_pred, dim=1)
    y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 4, 1, 2, 3)
    dice_numerator = 2 * torch.sum(y_true * y_pred, dim=axis) + epsilon
    dice_denominator = torch.sum(y_true, dim=axis) + torch.sum(y_pred, dim=axis) + epsilon
    dice_coef = torch.mean(dice_numerator / dice_denominator)

    return dice_coef


def dice_score(gt, pred, n_class):

    epsilon = 1.0e-6
    prediction = torch.log_softmax(pred, dim=1)

    dice_scores = np.zeros(n_class, dtype=torch.float32)
    for class_id in range(n_class):
        img_A = torch.eq(gt, class_id).int()
        img_B = torch.eq(prediction, class_id).int()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


def tv_index(gt, pred, smooth=0.00001, n_class=3, alpha=0.7, beta=0.3):
    batchSize = len(gt)
    prediction = torch.log_softmax(pred, dim=1)
    _, prediction = torch.max(prediction, dim=1)

    Tversky_score = torch.zeros((batchSize, n_class), dtype=torch.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(gt, prediction)):
        for class_id in range(n_class):
            label_gt = torch.eq(l_gt, class_id).int().flatten()
            label_pred = torch.eq(l_pred, class_id).int().flatten()

            TP = torch.sum(label_gt * label_pred)
            FP = torch.sum((1 - label_gt) * label_pred)
            FN = torch.sum(label_gt * (1 - label_pred))
            score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
            Tversky_score[batch_id, class_id] = score

    return torch.mean(Tversky_score)


def tv_coefficient(y_true, y_pred, num_classes=3, axis=(0, 2, 3, 4), alpha=0.7, beta=0.3,
                   epsilon=0.00001):

    y_pred = F.softmax(y_pred, dim=1)
    y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 4, 1, 2, 3)

    TP = torch.sum(y_true * y_pred, dim=axis) + epsilon
    FP = torch.sum((1 - y_true) * y_pred, dim=axis)
    FN = torch.sum(y_true * (1 - y_pred))
    tv_coef = TP / (TP + alpha * FP + beta * FN + epsilon)

    return torch.mean(tv_coef)

