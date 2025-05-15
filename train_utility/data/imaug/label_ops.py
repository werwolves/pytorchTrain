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
        from transformers import  LayoutLMv3Processor 
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-large", apply_ocr=False)
        self.label_list = label_list
        self.max_seq_length = kwargs.get("max_seq_length", 512)
    def rescale_bboxes(self, boxes, width, height):
        rescaled_boxes = []
        for box in boxes:
            x0, y0, x1, y1 = box
            rescaled_x0 = int((x0 / width) * 1000)
            rescaled_y0 = int((y0 / height) * 1000)
            rescaled_x1 = int((x1 / width) * 1000)
            rescaled_y1 = int((y1 / height) * 1000)
            rescaled_boxes.append([rescaled_x0, rescaled_y0, rescaled_x1, rescaled_y1])
        return rescaled_boxes   
    
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
            
        img_h, img_w = data['image'].shape[:2]   
        new_format_boxes_list = self.rescale_bboxes(new_format_boxes_list, img_w, img_h)
        # -----------------------------------------
        encoding = self.processor(data['image'], words_list, boxes=new_format_boxes_list, max_length=self.max_seq_length, 
                                  truncation=True, padding="max_length", return_tensors="pt")
        
        data["input_ids"] = encoding["input_ids"]
        data["bbox"] = encoding["bbox"]
        data["attention_mask"] = encoding["attention_mask"]
        data["pixel_values"] = encoding["pixel_values"]
        
        sequence_length = encoding["input_ids"].shape[1]
        if len(labels) < sequence_length:
            classes_list += [-100] * (sequence_length - len(labels))
        data["classes"] = torch.tensor([classes_list], dtype=torch.long)
        
        return data