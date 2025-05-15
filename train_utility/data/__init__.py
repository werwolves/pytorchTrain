import os
import sys
import copy
from torch.utils.data import DataLoader, DistributedSampler
from .SimpleDataSet import SimpleDataSet

__all__ = ['build_dataset', 'transform', 'create_dataset']


def build_dataset(config, mode, logger, **kwargs):
    """
    Build a dataset based on the provided dataset name.
    """
    config = copy.deepcopy(config)
    support_list = [
        'SimpleDataSet'
    ]
    dataset_name = config[mode]['dataset']['name']  # 'SimpleDataSet'
    assert dataset_name in support_list, \
        "The dataset {} is not in support list: {}".format(dataset_name, support_list)
     
    assert mode in ['Train', 'Eval', 'Test'], \
        "The mode {} is not in support list: {}".format(mode, ['Train', 'Eval', 'Test'])
    # 建立自己数据集的 dataset   
    dataset = eval(dataset_name)(config, mode, logger, **kwargs)  # SimpleDataSet(config, mode, logger, **kwargs)
    
    # 以下是 Dataloader 的参数设置
    loader_config = config[mode]['loader']
    shuffle = loader_config['shuffle'] 
    drop_last = loader_config['drop_last'] 
    batch_size = loader_config['batch_size']  # 1
    num_workers = loader_config['num_workers']  # 0
    
    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
        
    batch_sampler = None
    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    if len(dataset) == 0:
        raise Exception(
            "The dataset length is 0. Please check if the images path in {} is correct".format(
                config[mode]['dataset']['data_dir']))
    
    return data_loader