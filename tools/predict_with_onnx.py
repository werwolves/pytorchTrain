import os, sys, time
from collections import OrderedDict
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
import onnxruntime
import torch
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoModelForTokenClassification, AutoConfig,AutoTokenizer
# from train_utility.data.imaug.image_utils import RandomResizedCropAndInterpolationWithTwoPic,Compose
from torchvision import transforms

class Ser_model(object):
    """ 模型配置加载相关的 """
    def __init__(self, onnx_model_path, **kwargs):
       

        self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name_l  = [i.name for i in self.onnx_session.get_inputs() ]
        self.output_name_l = [i.name for i in self.onnx_session.get_outputs()]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
                                            "microsoft/layoutlmv3-base-chinese",
                                            tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
                                            cache_dir="./weights/layoutlmv3-base-chinese-new",
                                            use_fast=True,
                                            add_prefix_space=True)
        self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        
        
        ############# 标签处理相关的 begin #############
        label_list = ['TITLE','AUTHOR','PUBLIC','CALL_NO','LIB_NAME','OTHER-1']
        self.label2ids = {"O": 0}
        for index, label in enumerate(label_list):
            self.label2ids["B-" + label] = 2 * index + 1  # 1 , 3 , 5
            self.label2ids["I-" + label] = 2 * index + 2  # 2 , 4 , 6
        self.ids2label = { value:key.replace('B-','').replace("I-",'')  for key, value in self.label2ids.items() }
        self.common_transform = transforms.Compose([     
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
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

        raw_img_w,raw_img_h = img.size
        raw_pil_img = img.copy()
        
        ocr_infoes = self.ocr_engine(img)[0]

       
        text_ids_list = []
        bbox_list = []
        attention_mask_list = []
        token_type_ids_list = []
        raw_text_list = []
        raw_bbox_list = []
        # 需要记录每一个实体的位置
        entry_len_list = [0]
        for ocr_info in ocr_infoes:
            bbox, text, _ = ocr_info
            raw_text_list.append(text)
            raw_bbox_list.append(bbox)
            encode_res = self.tokenizer(text, truncation=True, add_special_tokens=False, return_attention_mask=True, return_token_type_ids=True)
            # --------------------------------- tokenizer ---------------------------------
            cur_text_ids = encode_res['input_ids'] 
            
            text_ids_list.extend(cur_text_ids)
            entry_len_list.append(len(text_ids_list))
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
        #                                 0                            
        input_ids_padding = [self.tokenizer.cls_token_id]  + [self.tokenizer.sep_token_id] +  text_ids_list + [self.tokenizer.pad_token_id] * difference
        #TODO:  input_ids_padding = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.pad_token_id] * difference + [self.tokenizer.sep_token_id]
        attention_mask_padding = [1] + attention_mask_list + [1] + [0] * difference 
        # bbox_padding = [[0,0,0,0]] + bbox + [[1000, 1000, 1000, 1000]] + [[0,0,0,0]] * difference 
        bbox_padding = [[0,0,0,0]] + bbox_list + [[0, 0, 0, 0]] + [[0,0,0,0]] * difference 
        # token_type_ids_padding = [0] + token_type_ids_list + [0] + [self.tokenizer.pad_token_type_id] * difference 
        
        
        # for_patches, _ = self.common_transform(img, augmentation=False)
        for_patches= self.common_transform(img)
        img = self.patch_transform(for_patches)
        inputs = {
            "input_ids": np.array([input_ids_padding], dtype=np.int64),
            "bbox": np.array([bbox_padding], dtype=np.int64),
            "attention_mask": np.array([attention_mask_padding],dtype=np.int64),
            "pixel_values": np.array(img.unsqueeze(0), dtype=np.float32)
        }
        ###### 增加 pytorch2onnx 的相关代码 begin ######
        t0 = time.time()
        output = self.onnx_session.run(self.output_name_l, inputs)[0]
        print(f"onnx run time:=========> {time.time()-t0}")
        output = np.argmax(output, axis=2)
        output = output[0] # .cpu().numpy() # [1, 512]
        real_ids_len = self.max_seq_length - difference
        pred_real_ids = output[2:real_ids_len]
        print("="*10)
        print(entry_len_list)
        print(f"text_list:{raw_text_list}")
        
        def most_common_element(lst):
            if not lst:
                return None  # 如果列表为空，返回 None
            return max(set(lst), key=lst.count)
        
        ###################################
        from PIL import Image, ImageDraw, ImageFont
        draw = ImageDraw.Draw(raw_pil_img)
        
        def draw_bbox(draw, bbox, text, color='red'):
            def find_center(bbox):
                x0, y0 = bbox[0]
                x1, y1 = bbox[1]
                x2, y2 = bbox[2]
                x3, y3 = bbox[3]
                
                x_center = (x0 + x1 + x2 + x3) / 4
                y_center = (y0 + y1 + y2 + y3) / 4
                
                return (x_center, y_center)
            draw.polygon([tuple(i) for i in bbox], outline='red')
            font_path = "/mnt/disk4/projects/expore/pytorchTrain/fonts/simfang.ttf"
            font_size = 20  # 设置字体大小
            font = ImageFont.truetype(font_path, font_size)
            draw.text(find_center(bbox), text, font=font,fill="black")
        ###################################
        for index, (start,end) in enumerate(zip(entry_len_list[:-1],entry_len_list[1:])):
            print("*"*10)
            entry_cls_id =  pred_real_ids[start:end] 
            print(f'entry_cls_id:{entry_cls_id}')
            class_name = most_common_element([self.ids2label[i] for i in entry_cls_id])
            ocr_text = raw_text_list[index]
            ocr_bbox = raw_bbox_list[index]
            print(f'ocr_text:{ocr_text},ocr_bbox:{ocr_bbox},class_name:{class_name}')
            draw_bbox(draw, ocr_bbox, class_name, color='black')
        raw_pil_img.save("output_res.jpg")
            
   

if __name__ == "__main__":
    # step1: 模型配置相关的
    import cv2
    from PIL import Image
    # im_path = r'/mnt/disk4/projects/expore/pytorchTrain/train_data/layout/book-spine-tot-part/n0101003-07-03_15.jpg'
    im_path = r'train_data/layout/book-spine-tot-part/n0101003-07-03_17.jpg'
    cv_img = cv2.imread(im_path)
    pil_im = Image.open(im_path)
    # step2: 模型加载
    weight_path = r'/mnt/disk4/projects/expore/pytorchTrain/test.onnx'
    model = Ser_model(weight_path)
    
    # step3: 数据（送入）模型，处理相关的
    model(pil_im)
    # step4: 模型输出后处理    
    
    
    