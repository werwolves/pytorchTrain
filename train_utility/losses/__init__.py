from .cls_loss import ClsLoss
import copy
__all__ = ['ClsLoss']
def build_loss(config):
    config = copy.deepcopy(config)  
    support_list = ['ClsLoss']
    module_name = config.pop('name')
    assert module_name in support_list, Exception('loss only support {}'.format(
        support_list))
    module_class = eval(module_name)(**config)
    return module_class