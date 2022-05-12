from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter



class TRANSEncoder(nn.Module):
    """Transformer layer"""

    def __init__(self, input_size):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=10, dim_feedforward=600, dropout=0)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input: Tensor) -> Tensor:
        x = self.trans(input)
        #print("input", input.shape)
        #print("x", x.shape)
        return x
