import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, inputs):
        x = inputs @ inputs.t()
        outputs = F.sigmoid(x)
        return outputs