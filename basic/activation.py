import torch
import torch.nn as nn


class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """

    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x

# 和下面的步骤是一样的意思
# def forward(self, x):
#
#         norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
#         p = torch.sigmoid(norm_x)
#         x = self.alpha * x.mul(1-p) + x.mul(p)
#
#         return x

# 这是一个自定义的激活函数，称为 Dice 激活函数，

def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function

    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            act_layer = Dice()
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(dim=1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer
# 这是一个用于构造激活函数层的函数，它接收一个激活函数名称或激活函数类作为输入参数，并返回相应的激活函数层。