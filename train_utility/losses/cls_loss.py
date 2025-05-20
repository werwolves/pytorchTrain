import torch.nn as nn

class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, predicts, batch):
        labels = batch.long()
        # loss = self.loss_func(predicts['res'], labels)
        batch, seq_len, num_classes = predicts.logits.shape
        loss = self.loss_func(predicts.logits.view(batch * seq_len, num_classes), labels.view(-1))
        return loss