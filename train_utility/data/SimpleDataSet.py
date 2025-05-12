import os,json
import numpy as np
from torch.utils.data import Dataset
from .imaug import create_operators, transform

class SimpleDataSet(Dataset):

    def __init__(self, config, mode, logger, **kwargs):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        
        # 整个工程的全局配置
        global_config = config['Global']
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
            np.random.shuffle(self.data_lines)
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
            with open(this_flie_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines.extend(lines)
        return lines
    
    
    def get_ext_data(self):
        ext_data_num = 0
        for ops in self.ops:
            if hasattr(ops, 'ext_data_num'):  # 此处的 ops 显然是一个对象 | 判断 ops 是否有 ext_data_num 属性
                ext_data_num = getattr(ops, 'ext_data_num') # 
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx] # 取出前面的一部分操作(这些操作是一些基础操作)用于额外数据的处理
        
        ext_data = []
        
        while len(ext_data) < ext_data_num:  # 如果扩展数据的数量小于要求的数量，则继续添加操作 
            # 这里的 ops 是一个列表，里面是一些数据处理的操作
            file_idx = self.data_idx_order_list[np.random.randint(0, len(self.data_lines)-1)] # 随机选择一个数据的索引
            data_info = self.data_lines[file_idx] # 读取数据的信息
            file_name, label_info = data_info.strip('\n').strip('\t') # 读取数据的文件名和标签信息
            label_info = json.loads(label_info)
            img_path = os.path.join(self.data_dir, file_name) # 读取数据的路径
            assert os.path.exists(img_path), f"Image path {img_path} does not exist!"
            data = {'img_path': img_path, 'label': label_info} # 读取数据的标签信息 
            # 为防止图像的路径中含有中文字符
            with open(img_path, 'rb') as f:
                img = f.read()
                data['image'] = img   
            data = transform(data, load_data_ops) # 对数据进行处理
                
            ext_data.append(data)
        return ext_data
        
        
        
    def __getitem__(self, index):
        """ 训练/验证/测试时，获取数据集中的一条数据 """
        file_idx = self.data_idx_order_list[index]
        data_info = self.data_lines[file_idx]
        
        # print("-=" * 20)
        # print("data_info: ", data_info)
        # res = data_info.strip('\n').split('\t')
        # print(f"rec:{res}")
        file_name, label_info = data_info.strip('\n').split('\t')
        label_info = json.loads(label_info)
        img_path = os.path.join(self.data_dir, file_name)
        assert os.path.exists(img_path), f"Image path {img_path} does not exist!"
        data = {'img_path': img_path, 'label': label_info}
        
        # 为防止图像的路径中含有中文字符
        with open(img_path, 'rb') as f:
            img = f.read()
            data['image'] = img
        # 上述读取的是一个正常的数据，下面是读取一个额外的扩充数据，用于数据增强
        data['ext_data'] = self.get_ext_data()
        data =  transform(data, self.ops)
        
        return data
    
    
    def __len__(self):
        return len(self.data_idx_order_list)