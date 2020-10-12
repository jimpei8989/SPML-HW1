from torch import nn
from torch.nn import functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return -(targets * F.log_softmax(outputs, dim=1)).sum() / outputs.shape[0]
