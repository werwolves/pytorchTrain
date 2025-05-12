# 很重要的模型基础类
import torch.nn as nn
from train_utility.modeling.backbones import build_backbone
from train_utility.modeling.necks import build_neck
from train_utility.modeling.heads import build_head

__all__ = [ 'BaseModel']


class BaseModel(nn.Module):
    def __init__(self,  config):
        super(BaseModel, self).__init__()
        
        # ---------- 有关 backbone 的配置 ----------
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            self.backbone = build_backbone(config['Backbone'])  # backbone 是一个模型
            try:
                in_channels = self.backbone.out_channels
            except:
                in_channels = 2
            
        # ---------- 有关 neck 的配置 ----------
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            self.neck = build_neck(config['Neck'])  # neck 是一个模型
            
        # ---------- 有关 head 的配置 ----------
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config['Head']['in_channels'] = in_channels
            self.head = build_head(config['Head'])
        
        
    def forward(self, x):
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        if self.use_head:
            x = self.head(x)
        
        return x
        