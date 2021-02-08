import torch

from torch.autograd import Function, Variable
from torch import nn
from torch.nn import functional as F

ALPHA = 0.3
BETA = 0.7

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class MultiTverskyLoss(nn.Module):
    """Computes the Tversky loss [1].
            Args:
                :param alpha: controls the penalty for false positives.
                :param beta: controls the penalty for false negatives.
                :param eps: added to the denominator for numerical stability.
            Returns:
                tversky_loss: the Tversky loss.
            Notes:
                alpha = beta = 0.5 => dice coeff
                alpha = beta = 1 => tanimoto coeff
                alpha + beta = 1 => F beta coeff
            References:
                [1]: https://arxiv.org/abs/1706.05721
        """

    def __init__(self, n_classes=3):
        super(MultiTverskyLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, eps=1, alpha=0.7, beta=0.3):
        batch_size = inputs.size(0)
        # flatten label and prediction tensors
        inputs = F.softmax(inputs, dim=1)
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).contiguous()

        # True Positives, False Positives & False Negatives
        TP = torch.sum(inputs * target, dim=(0, 2, 3, 4))
        FP = torch.sum((1 - target) * inputs, dim=(0, 2, 3, 4))
        FN = torch.sum(target * (1 - inputs), dim=(0, 2, 3, 4))

        Tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)

        score = 1.0 - Tversky / (float(batch_size) * float(self.n_classes))

        return torch.mean(score)


class tversky(nn.Module):
    def __init__(self, n_classes=3):
        super(tversky, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, eps=1, alpha=0.7, beta=0.3):

        inputs = torch.sigmoid(inputs)
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).contiguous()
        inputs = inputs.view(-1)
        target = target.view(-1)

        # True Positives, False Positives & False Negatives
        TP = torch.sum(inputs * target)
        FP = torch.sum((1 - target) * inputs)
        FN = torch.sum(target * (1 - inputs))

        Tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)

        score = 1.0 - Tversky / float(self.n_classes)

        return torch.mean(score)


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.00001
        batch_size = input.size(0)

        # input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        input = F.softmax(input, dim=1)
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).contiguous()
        # target = target[:, 1:] #ignore background
        inter = torch.sum(input * target, dim=(0, 2, 3, 4)) + smooth
        union = torch.sum(input ** 2, dim=(0, 2, 3, 4)) + torch.sum(target ** 2, dim=(0, 2, 3, 4)) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score
