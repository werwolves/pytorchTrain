import os
from torch.utils.data import Dataset


class SimpleDataSet(Dataset):

    def __init__(self, config, mode, logger, **kwargs):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        
        global_config = config['global']
        dataset_config = config[mode]['dataset']
        loader_config = dataset_config['loader']
        
        # TODO: 明天继续写 数据加载逻辑
        
    def __getitem__(self, index):
        pass
    
    
    def __len__(self):
        return 0