import copy

__all__ = ['build_metric']
from .cls_metric import ClsMetric
def build_metric(config):
    support_list = [
        "ClsMetric"
    ]

    config = copy.deepcopy(config)  # 深拷贝，避免修改原始配置
    assert config["name"] in support_list, f"metric {config['name']} not supported!"
    metric = eval(config["name"])(config)
    return metric