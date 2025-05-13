import torch
import numpy as np

class ClsLabelEncode:
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list
    def __call__(self, data):
        label = data['label']
        if label not in self.label_list:
            raise ValueError(f"Label {label} not in label list.")
        # 这里的 label_list 是一个列表，里面是一些标签的名称
        label = self.label_list.index(label) 
        data['label'] = label
        return data
    
class LayoutEncode:
    def __init__(self, label_list, **kwargs):
        from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Config,LayoutLMv3Processor,LayoutLMv3ImageProcessor 
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.label_list = label_list
    
    def __call__(self, data):
        # im, words, boxes = data['image'], data['words'], data['boxes']
        labels = data['label']
        
        words_list = []
        boxes_list = []
        classes_list = []
        for label in labels:
            words_list.append(label["transcription"].strip())
            boxes_list.append(label["points"])
            # TODO: boxes_list 是一个二维列表，需要将其转换为一维列表
            classes_list.append(self.label_list.index(label["class"].strip().upper()))
        # -----------------------------------------
        new_format_boxes_list = []
        for box in boxes_list:
            tem_np_box = np.array(box)
            left, top = min(tem_np_box[:,0]), min(tem_np_box[:,1])
            right, botton = max(tem_np_box[:,0]), max(tem_np_box[:,1])
            
            new_format_boxes_list.append([left, top, right, botton])
            
            
            
        # -----------------------------------------
        encoding = self.processor(data['image'], words_list, boxes=new_format_boxes_list, return_tensors="pt")
        
        data["input_ids"] = encoding["input_ids"]
        data["bbox"] = encoding["bbox"]
        data["attention_mask"] = encoding["attention_mask"]
        data["pixel_values"] = encoding["pixel_values"]
        data["classes"] = torch.tensor(classes_list, dtype=torch.long)
        return data