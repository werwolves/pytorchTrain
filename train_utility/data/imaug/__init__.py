
from 



# 各种 数据增强类/数据操作类 的实例
def create_operators(op_list, global_config=None):
    support_list = [
        
        
    ]
    assert isinstance(op_list, list), "The op_list should be list!"
    ops = []
    for op in op_list:
        assert isinstance(op, dict), "The op_list should be list of dict!"
        op_name = list(op.keys())[0]
        assert op_name in support_list, \
            "The op {} is not in support list: {}".format(op_name, support_list)
        param = {} if op[op_name] is None else op[op_name]
        
        operator = eval(op_name)(param)
        ops.append(operator)
        
    return ops