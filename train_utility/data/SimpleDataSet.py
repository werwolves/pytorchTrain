import os
from torch.utils.data import Dataset
from .imaug import create_operators

class SimpleDataSet(Dataset):

    def __init__(self, config, mode, logger, **kwargs):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        
        # 整个工程的全局配置
        global_config = config['global']
        # 训练/验证 数据集的配置
        dataset_config = config[mode]['dataset']
        # 训练/验证 数据加载器的配置
        loader_config = config[mode]['loader']
        self.do_shuffle = loader_config['shuffle']
        # 一般指的是 图像数据的目录
        self.data_dir = dataset_config['data_dir']
        # 标签 （一般是txt文件，里面内容的格式为 图像路径\t对应的标签）
        self.label_file_list = dataset_config['label_file_list']
        self.data_lines = self.read_label_file()
        if self.mode == 'train' and self.do_shuffle:
            self.shuffle(self.data_lines)
        # 数据的索引顺序
        self.data_idx_order_list = list(range(len(self.data_lines)))
        ## ---------------------- 数据处理的操作 ----------------------
        # 将所有的数据操作都放在一个列表中，后续可以直接使用这个列表进行数据处理
        self.ops = create_operators(dataset_config['transforms'], global_config)  #  正常的数据操作
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2) #  扩展操作的索引（一般是指用于数据增强要用到的操作!!!）
        
    def  read_label_file(self):
        """ 读取标签文件 """
        lines = []
        for this_flie_path  in self.label_file_list:
            with open(self.label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines.extend(lines)
        return lines
    
    
    def get_ext_data(self):
        ext_data_num = 0
        for ops in self.ops:
            if hasattr(ops, 'ext_data_num'):  # 此处的 ops 显然是一个对象
                ext_data_num = ops.ext_data_num
                break
            
            
            
            
        
        random.choice()
        
    def __getitem__(self, index):
        """ 训练/验证/测试时，获取数据集中的一条数据 """
        file_idx = self.data_idx_order_list[index]
        data_info = self.data_lines[file_idx]
        
        file_name, label_info = data_info.strip('\n').strip('\t')
        img_path = os.path.join(self.data_dir, file_name)
        assert os.path.exists(img_path), f"Image path {img_path} does not exist!"
        data = {'img_path': img_path, 'label': label_info}
        
        # 为防止图像的路径中含有中文字符
        with open(img_path, 'rb') as f:
            img = f.read()
            data['img'] = img
        # 上述读取的是一个正常的数据，下面是读取一个额外的扩充数据，用于数据增强
        data['ext_data'] = self.get_ext_data()
        
    
    def __len__(self):
        return len(self.data_idx_order_list)