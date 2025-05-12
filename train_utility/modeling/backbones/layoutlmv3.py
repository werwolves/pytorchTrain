import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

class LayoutLMV3(nn.Module):
    def __init__(self, config):
        config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base")
        config.num_labels = config.get("num_labels", 2)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", config=config, cache_dir="weights/layoutlmv3")   
        
    def forwards(self, inputs):
        outputs = self.model(**inputs)
        return outputs