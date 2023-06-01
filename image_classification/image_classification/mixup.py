import paddle
import paddle.nn as nn
import numpy as np

def mixup(alpha, num_classes, data, target):
    with paddle.no_grad():
        bs = data.shape[0]
        c = np.random.beta(alpha, alpha)

        perm = paddle.randperm(bs)

        md = c * data + (1-c) * data[perm, :]
        mt = c * target + (1-c) * target[perm, :]
        return md, mt

class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes

    def mixup_loader(self, loader):
        for input, target in loader:
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

class NLLMultiLabelSmooth(nn.Layer):
    def __init__(self, smoothing = 0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.astype('float32')
            target = target.astype('float32')
            logprobs = nn.functional.log_softmax(x, axis = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(axis=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return nn.functional.cross_entropy(x, target)

