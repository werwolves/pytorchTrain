import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification, AutoModelForTokenClassification, AutoConfig
from transformers import LayoutLMv3ForSequenceClassification

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