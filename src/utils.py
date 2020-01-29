import torch
from torch import nn


class CrossEntropyLossSoft(nn.Module):

    def __init__(self, weight=torch.Tensor([1, 1])):
        super(CrossEntropyLossSoft, self).__init__()
        self.weight = weight

    def forward(self, pred, soft_targets):
        logsoftmax = nn.LogSoftmax()

        return torch.mean(torch.sum(- soft_targets * self.weight * logsoftmax(pred), 1))


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

    return true_dist
