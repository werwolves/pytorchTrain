import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification, AutoModelForTokenClassification, AutoConfig
# from transformers import LayoutLMv3ForSequenceClassification
from .modeling_tools import LayoutLMv3ForSequenceClassification

class LayoutLMV3_ous(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-large")
        config.num_labels = kwargs.get("num_labels", 2)
        self.model1 = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", config=config, cache_dir="weights/layoutlmv3-large")   
        
    def forward(self, inputs):
        if 'labels' in inputs:
            labels = inputs.pop('labels')
            
            outputs = self.model1(**inputs, labels=labels)
        else:
            outputs = self.model1(**inputs)
        return outputs


class LayoutLMV3_zh(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base-chinese",num_labels=kwargs.get("num_labels", 2),
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base-chinese",config=config,
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        
    def forward(self, inputs):
        if 'labels' in inputs:
            labels = inputs.pop('labels')
            
            outputs = self.model(**inputs, labels=labels)
        else:
            outputs = self.model(**inputs)
        return outputs
    
class LayoutLMV3_seq_zh(nn.Module):
    """
    基于LayoutLMv3的中文序列分类模型

    使用预训练的中文LayoutLMv3模型进行文档级别的序列分类任务

    Args:
        **kwargs: 可变关键字参数
            - num_labels: 分类标签数量，默认为2
    """
    def __init__(self, **kwargs):
        super().__init__()
        config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base-chinese",num_labels=kwargs.get("num_labels", 2),
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base-chinese",config=config,
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        
    def forward(self, inputs):
        if 'labels' in inputs:
            labels = inputs.pop('labels')
            
            outputs = self.model(**inputs, labels=labels)
        else:
            outputs = self.model(**inputs)
        return outputs