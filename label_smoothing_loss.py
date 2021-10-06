import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, weight=None, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is not None:
            self.weight = weight.cuda()
        else:
            self.weight = None

    def forward(self, pred, target):
        pred = torch.hstack((torch.zeros_like(pred), pred))
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        if self.weight is not None:
            output = torch.mean(-true_dist * pred, dim=self.dim) * self.weight[target]
            # output = torch.sum(-true_dist * pred  * self.weight, dim=self.dim) / torch.sum(self.weight, dim=self.dim)
        else:
            output = torch.mean(-true_dist * pred, dim=self.dim)
        return output

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight=None, epsilon=0.1, reduce=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduce
        self.nll_loss = nn.NLLLoss(weight=weight, reduce=reduce)

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = self.nll_loss(log_preds, target)
        return linear_combination(loss / n, nll, self.epsilon)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=False, weight = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight

    def forward(self, inputs, targets):
        targets = targets.float()
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight = self.weight, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, weight = self.weight, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss