import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification, LayoutLMv3Processor

class LayoutLMV3_ous(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base")
        config.num_labels = kwargs.get("num_labels", 2)
        self.model1 = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", config=config, cache_dir="weights/layoutlmv3")   
        
    def forward(self, inputs):
        if 'labels' in inputs:
            labels = inputs.pop('labels')
            
            outputs = self.model1(**inputs, labels=labels)
        else:
            outputs = self.model1(**inputs)
        return outputs