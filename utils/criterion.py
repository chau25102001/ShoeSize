import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, label):
        pred = torch.softmax(pred, dim=1)
        label = label.type(pred.type())
        dims = (0,) + tuple(range(2, pred.ndimension()))
        tps = torch.sum(pred * label, dims)
        fps = torch.sum(pred * (1 - label), dims)
        fns = torch.sum((1 - pred) * label, dims)
        tversky = (tps / (tps + self.alpha * fps + self.beta * fns)).mean()

        return 1 - tversky


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLoss()(pred, mask)
    wfocal = (wfocal * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wfocal + wiou).mean()


class MyCriterion(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MyCriterion, self).__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, pred, label):
        focal = structure_loss(pred, label)
        tversky = self.tversky(pred, label)

        return focal, tversky
