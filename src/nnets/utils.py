import torch
from torch import nn
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE


class CrossEntropyLossSoft(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.weight = weight

    def forward(self, pred, soft_targets):
        logsoftmax = nn.LogSoftmax()
        if self.weight is not None:
            return torch.mean(torch.sum(- soft_targets * self.weight * logsoftmax(pred), 1))
        else:
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


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


def ece_score(y_true, y_prob, n_bins=10):
    ece = ECE(n_bins)
    ece_val = ece.measure(y_prob, y_true)

    return ece_val


def plot_reliability_diagram(y_true, y_prob, n_bins=10, title_suffix=''):
    diagram = ReliabilityDiagram(n_bins)
    diagram.plot(y_prob, y_true, title_suffix)

