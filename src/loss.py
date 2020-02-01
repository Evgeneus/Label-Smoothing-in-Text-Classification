import torch
from torch import nn


class CrossEntropyLossSoftECE(nn.Module):

    def __init__(self, weight=torch.Tensor([1, 1]), beta=0.001, bins=10):
        super(CrossEntropyLossSoftECE, self).__init__()
        self.weight = weight
        self.beta = beta
        self.bins = bins

    def _bucketize(self, X, bin_boundaries):
        result = torch.zeros_like(X, dtype=torch.int32)
        for boundary in bin_boundaries:
            result += (X > boundary).int()
        return result

    def _compute_ece(self, X, y):
        """
        Measure calibration by given predictions with confidence and the according ground truth.
        Assume binary predictions with y=1.

        Parameters
        ----------
        X : tensor with confidence values for each prediction.
            1-D for binary classification
        y : tensor with ground truth labels.

        Returns
        -------
        float
            Expected Calibration Error (ECE).
        """
        # get total number of samples
        num_samples = y.size()[0]
        prediction = torch.ones_like(X)

        # compute where prediction matches ground truth
        matched = prediction == y

        # create bin bounds
        bin_boundaries = torch.linspace(0.0, 1.0, self.bins + 1)

        # now calculate bin indices
        # this function gives the index for the upper bound of the according bin
        # for each sample. Thus, decrease by 1 to get the bin index
        current_indices = self._bucketize(X, bin_boundaries) - 1

        current_indices[current_indices == -1] = 0
        current_indices[current_indices == self.bins] = self.bins - 1

        ece = 0.0
        # mean accuracy is new confidence in each bin
        for bin in range(self.bins):
            bin_confidence = X[current_indices == bin]
            bin_matched = matched[current_indices == bin]
            if bin_confidence.size()[0] > 0:
                bin_weight = float(bin_confidence.size()[0]) / float(num_samples)
                ece += bin_weight * torch.abs(torch.mean(bin_matched.float()) - torch.mean(bin_confidence))

        return ece

    def forward(self, pred, soft_targets, hard_target):
        logsoftmax = nn.LogSoftmax()
        cross_entropy_soft = torch.mean(torch.sum(- soft_targets * self.weight * logsoftmax(pred), 1))
        ece = self._compute_ece(torch.sigmoid(pred)[:, 1], hard_target)
        loss = cross_entropy_soft + self.beta * ece

        return loss
