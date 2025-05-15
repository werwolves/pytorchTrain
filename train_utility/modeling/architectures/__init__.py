import copy
import importlib
from .base_model import BaseModel

__all__ = ['build_model']

def build_model(config):
    config = copy.deepcopy(config)  # 深拷贝，避免修改原始配置
    arch = BaseModel(config)
    return arch
