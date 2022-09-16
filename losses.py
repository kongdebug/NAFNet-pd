import paddle
import paddle.nn as nn
import numpy as np



def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, paddle.Variable of paddle.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = paddle.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = paddle.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(paddle.sum(paddle.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Layer):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = paddle.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = paddle.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = paddle.sum(paddle.sqrt(diff * diff + self.eps))
        loss = paddle.mean(paddle.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = paddle.to_tensor(np.array([65.481, 128.553, 24.966])).reshape([1, 3, 1, 1])

    def forward(self, pred, target):
        if self.toY:
            pred = (pred * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.
            target = (target * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.

            pred, target = pred / 255., target / 255.
            pass

        return self.loss_weight * self.scale * paddle.log(((pred - target) ** 2).mean(axis=[1, 2, 3]) + 1e-8).mean()
