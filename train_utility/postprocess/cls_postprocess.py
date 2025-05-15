import torch

__all__ = ["ClsPostProcess"]

class ClsPostProcess:
   """ 在推理后，对模型输出进行后处理"""
   def __init__(self,label_list=None, **kwargs):
      super().__init__(**kwargs)
      self.label_list = label_list  # 标签列表 eg: ['dog', 'cat']
      
   def __call__(self, preds, batch=None, *args, **kwargs):
      if 'res' in preds:
         preds = preds['res']
      if isinstance(preds, torch.Tensor):
         preds = preds.detach().cpu().numpy()
      preds_idx = preds.argmax(axis=1)
      
      if self.label_list is not None:
         self.label_list = {i:i for i in range(preds.shape[-1])} # {0:0, 1:1}
      decode_out = [ (self.label_list[idx], preds[index][idx])  for index,idx in enumerate(preds_idx) ]
      
      if batch is None:
         return decode_out
       
      return decode_out, [(self.label_list[i], 1.0) for i in batch[1].detach().cpu().numpy() ]
      
      
      