__all__ = ["ClsMetric"]


class ClsMetric:
    def __init__(self, main_indicator='acc' , **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-8
        self.reset()
        
    def __call__(self, pred_label, *args, **kwds):
        preds, labels = pred_label
    
        for (pred, pred_conf),(label, _) in zip(preds, labels):
            self.tot_num += 1
            if pred == label:
                self.correct_num += 1
        return {"acc": self.correct_num / (self.tot_num + self.eps)}
    
    def get_metric(self):
        acc =  {"acc": self.correct_num / (self.tot_num + self.eps)}
        self.reset()
        return acc
    
    def reset(self):
        self.correct_num = 0
        self.tot_num = 0