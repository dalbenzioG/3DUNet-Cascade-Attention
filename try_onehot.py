import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target, smooth=1, alpha=ALPHA, beta=BETA):
        batch_size = input.size(0)
        # flatten label and prediction tensors
        # input = F.softmax(input[:, self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        # target = target[:, self.class_ids, :]
        # target[target != self.ignore_index] = target
        # target = target * (target != self.ignore_index).long()

        # True Positives, False Positives & False Negatives
        TP = (input * target).sum()
        FP = ((1 - target) * input).sum()
        FN = (target * (1 - input)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        score = 1.0 - Tversky / (float(batch_size) * float(self.n_classes))

        return score


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).to(device)  # cuda()

    def forward(self, X_in):
        print('X_in before{}'.format(X_in.shape))
        n_dim = X_in.dim()
        print('n_dim {}'. format(n_dim))
        output_size = X_in.size() + torch.Size([self.depth])
        print('output_size {}'.format(output_size))
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        print('X_in {}'.format(X_in.shape))
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        print('out {}'.format(out.shape))
        out1 = out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()
        print('out1 {}'.format(out1.shape))
        return out1

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


# if __name__ == '__main__':

# depth = 3
# batch_size = 2
# encoder = One_Hot(depth=depth).forward
# y = Variable(torch.LongTensor(batch_size, 1, 1, 2, 2).random_() % depth).cuda()
# print('y {}'.format(y.shape))
# print(y)# 4 classes,1x3x3 img
# y_onehot = encoder(y)
# print('y_onehot {}'.format(y_onehot.shape))
# print(y_onehot)
# x = Variable(torch.randn(y_onehot.size()).float()).cuda()

x = Variable(torch.randn(1, 3, 256, 256, 256).float()).cuda()
x1 = Variable(torch.randn(1, 256, 256, 256).float()).cuda()
target = x1.to(torch.int64)
target[target < 0] = 0
target[target >= 3] = 0

dicemetric = SoftDiceLoss(n_classes=3)
dicemetric(x, target)





