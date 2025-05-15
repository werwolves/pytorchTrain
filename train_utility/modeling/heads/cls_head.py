import torch
import torch.nn.functional as F
from torch import nn


class ClsHead(nn.Module):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, class_dim)
        self.dropout = nn.Dropout(p=0.7)
        
    def forward(self, x, data=None):  # x.shape = torch.Size([1, 1024, 14, 14])
        x = self.pool(x)              # x.shape = torch.Size([1, 1024, 1, 1])
        x = torch.reshape(x, shape=[x.shape[0], x.shape[1]]) # x.shape = torch.Size([1, 1024])
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return {'res': x}