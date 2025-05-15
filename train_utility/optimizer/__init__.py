import copy
import torch

__all__ = ['build_optimizer']

def build_optimizer(optim_config, lr_scheduler_config ,model):
    from . import lr
    
    # step1: 优化器相关的配置
    optim_config = copy.deepcopy(optim_config)
    train_parameters = filter(lambda p: p.requires_grad, model.parameters() )
    optim_name = optim_config.pop('name')
    optim = getattr(torch.optim, optim_name)(train_parameters, **optim_config)
    
    # step2: 学习率相关的配置
    lr_scheduler_config = copy.deepcopy(lr_scheduler_config)
    lr_scheduler_name = lr_scheduler_config.pop('name')
    lr_scheduler = getattr(lr, lr_scheduler_name)(**lr_scheduler_config)(optim)
    
    return optim, lr_scheduler


