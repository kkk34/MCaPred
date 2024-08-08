# @Author  : hdm
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class BinaryClassificationLoss(nn.Module):
    """Automatically weighted binary classification loss"""
    def __init__(self):
        super(BinaryClassificationLoss, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, requires_grad=True))
        # self.num_tasks =  num_tasks

    def forward(self, loss):
        weighted_loss = self.weight * loss + torch.log(1 + self.weight ** 2)
        return weighted_loss
