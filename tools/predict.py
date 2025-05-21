import os,sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))



import torch
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoModelForTokenClassification, AutoConfig
from train_utility.data.imaug.image_utils import RandomResizedCropAndInterpolationWithTwoPic,Compose
from torchvision import transforms

class Ser_model(object):
    """ 模型配置加载相关的 """
    def __init__(self, weight_path, **kwargs):
        # TODO: 模型加载还可以继续优化
        config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base-chinese",num_labels=13,
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base-chinese",config=config,
                                            cache_dir="./weights/layoutlmv3-base-chinese-new")
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        
        
        self.common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=224
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.5, 0.5, 0.5)),
                std=torch.tensor((0.5, 0.5, 0.5)))
        ])
        
    def rescale_bboxes(self, box, width, height):
        x0, y0, x1, y1 = box
        rescaled_x0 = int((x0 / width) * 1000)
        rescaled_y0 = int((y0 / height) * 1000)
        rescaled_x1 = int((x1 / width) * 1000)
        rescaled_y1 = int((y1 / height) * 1000)
        return [rescaled_x0, rescaled_y0, rescaled_x1, rescaled_y1] 
    
    def __call__(self, img):
        raw_img_w, raw_img_h = img.shape[:2]
        ocr_infoes = self.ocr_engine(img)
       
       
        text_ids_list = []
        bbox_list = []
        attention_mask_list = []
        token_type_ids_list = []
        for ocr_info in ocr_infoes:
            bbox, text, _ = ocr_info
            encode_res = self.tokenizer(text, truncation=True, add_special_tokens=False, return_attention_mask=True, return_token_type_ids=True)
            # --------------------------------- tokenizer ---------------------------------
            cur_text_ids = encode_res['input_ids'] 
            text_ids_list.extend(cur_text_ids)
            # --------------------------------- bbox --------------------------------------
            tem_np_box = np.array(bbox)
            left, top = min(tem_np_box[:,0]), min(tem_np_box[:,1])
            right, bottom = max(tem_np_box[:,0]), max(tem_np_box[:,1])
            cur_text_box = self.rescale_bboxes([left,top,right,bottom], raw_img_w, raw_img_h)
            cur_text_box = [cur_text_box] * len(cur_text_ids)
            bbox_list.extend(cur_text_box)
            # --------------------------------- attention_mask ---------------------------
            attention_mask = [1] * len(cur_text_ids)
            attention_mask_list.extend(attention_mask)
            # --------------------------------- token_type_ids --------------------------
            token_type_ids = encode_res["token_type_ids"]
            token_type_ids_list.extend(token_type_ids)
        # --------------------------------- padding ---------------------------------
        difference = self.max_seq_length - len(text_ids_list) - 2
        input_ids_padding = [self.tokenizer.cls_token_id]  + [self.tokenizer.sep_token_id] +  text_ids_list + [self.tokenizer.pad_token_id] * difference
        attention_mask_padding = [1] + attention_mask_list + [1] + [0] * difference 
        # bbox_padding = [[0,0,0,0]] + bbox + [[1000, 1000, 1000, 1000]] + [[0,0,0,0]] * difference 
        bbox_padding = [[0,0,0,0]] + bbox_list + [[0, 0, 0, 0]] + [[0,0,0,0]] * difference 
        token_type_ids_padding = [0] + token_type_ids_list + [0] + [self.tokenizer.pad_token_type_id] * difference 
        
        
        for_patches, _ = self.common_transform(img, augmentation=False)
        img = self.patch_transform(for_patches)
        inputs = {
            "input_ids": torch.tensor([input_ids_padding], dtype=torch.long).to(self.model.device),
            "bbox": torch.tensor([bbox_padding], dtype=torch.float).to(self.model.device),
            "attention_mask": torch.tensor([attention_mask_padding], dtype=torch.long).to(self.model.device),
            "pixel_values": torch.tensor([img], dtype=torch.float).to(self.model.device)
        }
        
        out = self.model(
            **inputs
        )
        output = out.logits
        output = torch.argmax(output, dim=2)
        output = output[0].cpu().numpy() # [1, 512]
        ###################
        # TODO: 这里需要对 512 长度的类别做处理
        ###################
        

        

if __name__ == "__main__":
    # step1: 模型配置相关的
    import cv2
    im_path = r'D:\projects\RFID\pytorchTrain\train_data\layout\book-spine-tot\n0101003-07-03_15.jpg'
    cv_img = cv2.imread(im_path)
    # step2: 模型加载
    model = Ser_model()
    
    # step3: 数据（送入）模型，处理相关的
    model(cv_img)
    
    # step4: 模型输出后处理    
    
    
    