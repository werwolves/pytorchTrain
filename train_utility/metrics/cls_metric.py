__all__ = ["ClsMetric"]


class ClsMetric:
    def __init__(self, num_classes):
        self.reset()
    def __call__(self, pred_label, *args, **kwds):
        preds, labels = pred_label
        preds = preds.argmax(dim=1)
        self.correct_num += (preds == labels).sum().item()
        self.tot_num += len(labels)
    
    def reset(self):
        self.correct_num = 0
        self.tot_num = 0