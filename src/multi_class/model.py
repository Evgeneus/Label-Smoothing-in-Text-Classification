import torch


class MLP1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs