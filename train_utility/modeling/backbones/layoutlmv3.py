import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

class LayoutLMV3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base")
        config.num_labels = kwargs.get("num_labels", 2)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", config=config, cache_dir="weights/layoutlmv3")   
        
    def forwards(self, inputs):
        outputs = self.model(**inputs)
        return outputs