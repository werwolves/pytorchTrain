import math,copy
from functools import partial
from torch.optim import lr_scheduler

class StepLR:
    def __init__(self, 
                 warmup_epoch=0, 
                 gamma=0.9,
                 last_epoch=-1,
                 **kwargs   
                 ):
        super(StepLR, self).__init__()
        self.lr_config = copy.deepcopy(kwargs)
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.gamma = gamma
        self.initial_lr = self.lr_config.pop('lr')  # 初始学习率 
        self.warpup_init_lr = self.lr_config.get('warpup_init_lr', 0)  # 初始学习率 (同时也是warmup的目标学习率)
        self.warpup_target_lr = self.lr_config.get('warpup_target_lr', self.initial_lr)
        
    def __call__(self, optimizer):
        #  每次执行 lr_scheduler_.step() 时，last_epoch =+ 1, 并将 last_epoch 传入 lambda_func
        lr_scheduler_ = lr_scheduler.LambdaLR(optimizer, self.lambda_func, last_epoch=self.last_epoch)
        return lr_scheduler_
    
    def lambda_func(self, epoch):
        """ 最终的学习率是 原始学习率 * 该函数的返回值"""
        if epoch < self.warmup_epoch:
            return (self.warpup_init_lr + (self.warpup_target_lr - self.warpup_init_lr) * (epoch) / max(1,self.warmup_epoch-1)) / self.warpup_target_lr
        else:
            return self.gamma ** epoch
        
        
        
if __name__ == '__main__':
    lr_config = {
        'lr': 0.001,
        'warmup_epoch': 2,
        'warpup_init_lr': 0.0001,
        # 'warpup_target_lr': 0.001,
        "NUM_EPOCHS": 100
    }
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import matplotlib.pyplot as plt

    def schedular_lr(optimizer, schedular):
        lr_history = []
        epoch_history = []
        """optimizer的更新在scheduler更新的前面"""
        for epoch in range(lr_config["NUM_EPOCHS"]):
            optimizer.step() # 更新参数
            lr_history.append(optimizer.param_groups[0]['lr'])
            schedular.step() # 调整学习率
        plt.plot(lr_history)
        plt.show()
        return lr_history
    
    model = torch.nn.Linear(10, 2)  # 示例模型
    optimizer = optim.SGD(model.parameters(), lr=lr_config['lr'])
    lr_scheduler_ = StepLR(**lr_config)(optimizer)
    

    
    lr_l = schedular_lr(optimizer, lr_scheduler_)
    print(lr_l)
