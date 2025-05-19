import os
import sys

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from train_utility.engine.trainer import Trainer
# from train_utility.engine.callbacks import LossHistory
from utility import ArgsParser, Config

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=True,
        help="Whether to perform evaluation in train")
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    print("config:",FLAGS.config)
    print("config-var:",Config(FLAGS.config).cfg)  # 可以从配置文件中成功的读取出配置参数!!!
    # 这是一个字典类型的配置参数
    config = Config(FLAGS.config).cfg  
    
    trainer = Trainer(config)
    trainer.train(config['Global']['epoch_num'])  # 训练模型的轮数

if __name__ == "__main__":
    main()