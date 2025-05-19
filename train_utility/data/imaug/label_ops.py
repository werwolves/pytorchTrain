import torch,random 
from torchvision import transforms
import numpy as np
from PIL import Image
from .image_utils import RandomResizedCropAndInterpolationWithTwoPic,Compose


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
    
    
class LayoutEncode_zh:
    def __init__(self, label_list, **kwargs):
        from transformers import  AutoTokenizer 
        import torch.nn as nn
        self.pad_token_label_id = nn.CrossEntropyLoss(ignore_index=-100).ignore_index
        
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
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(
                                                    "microsoft/layoutlmv3-base-chinese",
                                                    tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
                                                    cache_dir="./weights/layoutlmv3-base-chinese-new",
                                                    use_fast=True,
                                                    add_prefix_space=True,
                                                )
        # self.label_list = label_list # ['TITLE','AUTHOR','PUBLIC','CALL_NO','LIB_NAME','OTHER-1']
        """
        需要将每一个 label 都转换为  B-xxx, I-xxx , 并加上一个额外的标签 'O'
        """
        self.label2ids = {"O": 0}
        for index, label in enumerate(label_list):
            self.label2ids["B-" + label] = 2 * index + 1  # 1 , 3 , 5
            self.label2ids["I-" + label] = 2 * index + 2  # 2 , 4 , 6

        
        
        
        self.max_seq_length = kwargs.get("max_seq_length", 512)
        
        
        
        
    def random_chinese_characters(num_chars):
        # 定义汉字的 Unicode 范围
        basic_range = (0x4E00, 0x9FFF)  # 基本汉字
        extension_a_range = (0x3400, 0x4DBF)  # 扩展 A
        extension_b_range = (0x20000, 0x2A6DF)  # 扩展 B

        # 合并所有范围
        ranges = [basic_range, extension_a_range, extension_b_range]

        result = []
        for _ in range(num_chars):
            # 随机选择一个范围
            chosen_range = random.choice(ranges)
            # 在选定的范围内随机生成一个 Unicode 码点
            code_point = random.randint(chosen_range[0], chosen_range[1])
            # 将 Unicode 码点转换为字符
            char = chr(code_point)
            result.append(char)

        return ''.join(result)  
    
      
    def rescale_bboxes(self, box, width, height):
        x0, y0, x1, y1 = box
        rescaled_x0 = int((x0 / width) * 1000)
        rescaled_y0 = int((y0 / height) * 1000)
        rescaled_x1 = int((x1 / width) * 1000)
        rescaled_y1 = int((y1 / height) * 1000)
        return [rescaled_x0, rescaled_y0, rescaled_x1, rescaled_y1]
     
    def random_chinese_characters(num_chars):
        # 定义汉字的 Unicode 范围
        basic_range = (0x4E00, 0x9FFF)  # 基本汉字
        extension_a_range = (0x3400, 0x4DBF)  # 扩展 A
        extension_b_range = (0x20000, 0x2A6DF)  # 扩展 B

        # 合并所有范围
        ranges = [basic_range, extension_a_range, extension_b_range]

        result = []
        for _ in range(num_chars):
            # 随机选择一个范围
            chosen_range = random.choice(ranges)
            # 在选定的范围内随机生成一个 Unicode 码点
            code_point = random.randint(chosen_range[0], chosen_range[1])
            # 将 Unicode 码点转换为字符
            char = chr(code_point)
            result.append(char)

        return ''.join(result)
    
    def __call__(self, data):
        # --- 这里是准对一篇文档的 数据处理，在设立专指  一张书脊图像上的数据处理
        
        # im, words, boxes = data['image'], data['words'], data['boxes']
        data_labels = data['label']
        
        attention_mask = []
        token_type_ids = []
        input_ids = []
        bbox = []
        labels = []
        for label in data_labels:
            # words_list.append(label["transcription"].strip())
            # boxes_list.append(label["points"])
            # TODO: boxes_list 是一个二维列表，需要将其转换为一维列表
            # classes_list.append(self.label_list.index(label["class"].strip().upper()))
            
            # -------------------------------- 文字处理 ------------------------------------
            """  将文字 转换为 token_id
            eg:  cur_text_ids = [13129, 13663, 75224]   ===> ('我', '是一个', '中国人')
            """
            if label["transcription"] == '':
                # label["transcription"] = self.random_chinese_characters(random.randint(1,5))
                continue
            encode_res = self.tokenizer(label["transcription"].strip(), truncation=True, add_special_tokens=False, return_attention_mask=True, return_token_type_ids=True)
            cur_text_ids = encode_res['input_ids'] 
            
            # -------------------------------- 类别处理 ------------------------------------
            """  每个 token 都有一个 label
            eg: cur_text_label = [3,2,2]  ===> ['B-TITLE', 'I-TITLE', 'I-TITLE']
            """
            cur_text_label = label["class"].strip().upper()
            # 需要让所有 token 共享 label
            if cur_text_label == "OTHER":
                cur_text_label = ["O"] * len(cur_text_ids)
                for i in cur_text_ids:
                    cur_text_label[i] = self.label2ids['O']
            else:
                try:
                    cur_text_label = [cur_text_label] * len(cur_text_ids)
                    cur_text_label[0] = self.label2ids['B-' + cur_text_label[0] ]
                    for i in range(1, len(cur_text_ids)):
                        cur_text_label[i] = self.label2ids['I-' + cur_text_label[i] ]
                except:
                    print("lolol")
            # --------------------------------- bbox 处理 ------------------------------------------------
            """ 每个 token 都有一个 bbox
            eg: cur_text_box = [ [left1, top1, right1, botton1],[left1, top1, right1, botton1],[left1, top1, right1, botton1]  ] 
            """
            cur_text_box = label["points"]
            cur_text_box = np.array(cur_text_box)
            
            ########################## 图像处理 ##############################
            
            
            with open(data["img_path"], 'rb') as f:
                img = Image.open(f)
                data["image"] = img.convert('RGB')
            img_w, img_h  = data['image'].size
            
            for_patches, _ = self.common_transform(data['image'], augmentation=False)
            patch = self.patch_transform(for_patches)
            data["image"] = patch
            
            
            ##################################################################
            
            
            
            # 即使是多边形，也获取器其外接矩形
            left,     top = min(cur_text_box[:,0]), min(cur_text_box[:,1])
            right, botton = max(cur_text_box[:,0]), max(cur_text_box[:,1])
                
            # img_h, img_w = data['image'].shape[1:]  
            cur_text_box = self.rescale_bboxes([left,top,right,botton], img_w, img_h) # [left, top, right, botton]
            cur_text_box = [cur_text_box] * len(cur_text_ids)
            

            ###### 该图像（文档）每个条目的 文字数量或token数量并不是等长的
            # 1. 将太短 < self.max_seq_length 的token加上padding
            # difference = self.max_seq_length - len(cur_text_ids)
            # 1的位置应该 attention, 0的位置应该忽略掉!
            
            # 将一个图像文档中的所有token拼起来----拼起来之后，再看长度是否超标！ 这里的逻辑是每个文本都看是否超标，显然该逻辑有问题!!!
            # attention_mask.extend([1] * len(cur_text_ids) + [0] * difference)
            # input_ids.extend(cur_text_ids + [self.tokenizer.pad_token_id] * difference)
            # token_type_ids.extend(encode_res["token_type_ids"] + [self.tokenizer.pad_token_type_id] * difference)
            # bbox.extend(cur_text_box + [[0,0,0,0]] * difference)
            # labels.extend(cur_text_label + [self.pad_token_label_id] * difference)
            
            
            ######################### 重新编写该逻辑
            
            attention_mask.extend([1] * len(cur_text_ids))
            input_ids.extend(cur_text_ids)
            token_type_ids.extend(encode_res["token_type_ids"])
            bbox.extend(cur_text_box)
            labels.extend(cur_text_label)
        # -------------------------- 对该文档中的所有token进行拼接 begin
        difference = self.max_seq_length - len(input_ids) - 2
        input_ids_padding = [self.tokenizer.cls_token_id] +  input_ids + [self.tokenizer.pad_token_id] * difference + [self.tokenizer.sep_token_id]
        attention_mask_padding = [0] + attention_mask + [0] * difference + [0]
        bbox_padding = [[0,0,0,0]] + bbox + [[0,0,0,0]] * difference + [[1000, 1000, 1000, 1000]]
        token_type_ids_padding = [0] + token_type_ids + [self.tokenizer.pad_token_type_id] * difference + [0]
        labels_padding = [-100] + labels + [self.pad_token_label_id] * difference  + [-100]
        
        # TODO: 当文档中的数据过长时，需要添加额外的处理逻辑
        
        
        
        
        
        # -------------------------- 对该文档中的所有token进行拼接 end    
        data["attention_mask"] = attention_mask_padding
        data["token_type_ids"] = token_type_ids_padding
        data["input_ids"] = input_ids_padding
        data["bbox"] = bbox_padding
        data["labels"] = labels_padding
        # data["image"] = data["image"][None, :, :, :]  # 这里是为了将数据转换为 [1,3,224,224] 的格式
        for key in data:
            if key in [
                        "input_ids",
                        "labels",
                        "token_type_ids",
                        "bbox",
                        "attention_mask",
                        ]:
                data[key] = np.array(data[key], dtype="int64")

                            
        return data
        
        
        
        #TODO: # 2. 将太长 > self.max_seq_length 的token进行截断
                     
           
            
            
            
            
            
            
            
            
            
            
            
        # -----------------------------------------
        # new_format_boxes_list = []
        # for box in boxes_list:
        #     tem_np_box = np.array(box)
        #     left, top = min(tem_np_box[:,0]), min(tem_np_box[:,1])
        #     right, botton = max(tem_np_box[:,0]), max(tem_np_box[:,1])
            
        #     new_format_boxes_list.append([left, top, right, botton])
            
        # img_h, img_w = data['image'].shape[:2]   
        # new_format_boxes_list = self.rescale_bboxes(new_format_boxes_list, img_w, img_h)
        # # -----------------------------------------
        # encoding = self.processor(data['image'], words_list, boxes=new_format_boxes_list, max_length=self.max_seq_length, 
        #                         truncation=True, padding="max_length", return_tensors="pt")
        
        # data["input_ids"] = encoding["input_ids"]
        # data["bbox"] = encoding["bbox"]
        # data["attention_mask"] = encoding["attention_mask"]
        # data["pixel_values"] = encoding["pixel_values"]
        
        # sequence_length = encoding["input_ids"].shape[1]
        # if len(labels) < sequence_length:
        #     classes_list += [-100] * (sequence_length - len(labels))
        # data["classes"] = torch.tensor([classes_list], dtype=torch.long)
        
        return data