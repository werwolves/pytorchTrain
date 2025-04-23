import copy
"""
各种模型的后处理
"""

__all__ = ['build_post_process']
from .cls_postprocess import ClsPostProcess


def build_post_process(config):
    support_list = [  
        ClsPostProcess
    ]
    
    assert config["name"] in support_list, f"post_process {config['name']} not supported!"
    config = copy.deepcopy(config)  # 深拷贝，避免修改原始配置
    module_name = config.pop('name')
    return eval(module_name)(**config)