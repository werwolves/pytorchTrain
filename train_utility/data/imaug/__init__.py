
from .iaa_augment import IaaAugmenter
from .operators import DecodeImage, KeepKeys
from .label_ops import ClsLabelEncode
from .rec_img_aug import BaseDataAugmentation, ClsResizeImg
from .randaugment import RandAugment
def transform(data, ops=None):
    """
    对数据进行处理的函数
    :param data: 数据字典
    :param ops: 操作列表
    :return: 处理后的数据
    """
    if ops is None:
        return data
    
    for op in ops:
        data = op(data)
    
    return data


# 各种 数据增强类/数据操作类 的实例
def create_operators(op_list, global_config=None):
    support_list = [
        "IaaAugmenter", "DecodeImage", 
        "ClsLabelEncode", "RandAugment",
        "BaseDataAugmentation", "ClsResizeImg",
        "KeepKeys"
    ]
    assert isinstance(op_list, list), "The op_list should be list!"
    ops = []
    for op in op_list:
        assert isinstance(op, dict), "The op_list should be list of dict!"
        op_name = list(op.keys())[0]
        print("=" * 20)
        print("op_name: ", op_name)
        assert op_name in support_list, \
            "The op {} is not in support list: {}".format(op_name, support_list)
        param = {} if op[op_name] is None else op[op_name]
        try:
             operator = eval(op_name)(**param)
        except Exception as e:
            print(f"Error in creating operator {op_name}: {e}")
            continue
        ops.append(operator)
        
    return ops