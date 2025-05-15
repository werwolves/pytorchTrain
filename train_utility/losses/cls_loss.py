import torch.nn as nn

class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, predicts, batch):
        labels = batch.long()
        loss = self.loss_func(predicts['res'], labels)
        return loss