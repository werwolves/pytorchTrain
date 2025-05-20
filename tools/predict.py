import torch
from rapidocr_onnxruntime import RapidOCR
from transformers import AutoModelForTokenClassification, AutoConfig

class model:
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
        
    def __call__(self, ocr_info):
        pass




if __name__ == "__main__":
    # step1: 模型配置相关的
    
    # step2: 模型加载
    model = model()
    
    # step3: 数据（送入）模型，处理相关的
    
    # step4: 模型输出后处理    
    
    
    