import numpy as np
import torch, os
import torch.nn as nn
from tqdm import tqdm
from train_utility.modeling.architectures import build_model
from train_utility.engine.callbacks import LossHistory

__all__ = ['Trainer']


class Trainer:
    def __init__(self, config):
        self.model = build_model(config['Architecture'])
        self.train_data_loader = config.get('train_data_loader', None) 
        self.val_data_loader = config.get('val_data_loader', None) 
        self.loss_fn = config.get('loss', None)
        self.optimizer = config.get('optimizer', None)
        # 这轮模型 训练的epoch数
        self.epoch_num = config.get('epoch_num', None)
        # 是否采用半精度训练
        self.is_fp16 = config.get('is_fp16', False)
        # 在训练的过程中是否采用梯度裁剪
        self.is_clip_grad = config.get('is_clip_grad', False)
        # 保存模型的路径
        self.model_save_dir = config["Global"].get('output_dir', None)
        # 训练模型所使用的设备
        self.device = config.get('device', None)
        # 记录训练过程的
        # self.loss_history = config.get('loss_history', None)
        self.loss_history = LossHistory(self.model_save_dir, self.model, config['Global']['image_shape'][1:])
        
        # assert self.check_attribute(), "模型训练所需的参数未设置完整！"
        self.model.to(self.device)  # 将模型移动到指定设备上
        
        
        
    def check_attribute(self):
        # 获取当前类实例的所有属性及其值，以字典形式返回 {'model': <model>, 'train_data_loader': <train_loader>, ...}
        attributes = vars(self) 
        return all(value is not None for value in attributes.values())
    
    def fit_one_epoch(self, cur_epoch):
        """ 模型训练一个epoch的过程 """
        # 记录训练过程中的效果，比如：损失和准确率
        train_loss, val_loss = 0.0, 0.0
        train_accuracy, val_accuracy = 0.0, 0.0
        # 记录训练过程中的进程条
        print("Start training...")
        train_pbar = tqdm(total=len(self.train_data_loader), desc="Training epoch:{cur_epoch}/{self.epoch_num}",  postfix=dict, mininterval=0.3)
        # 
        
        # -------------------------------------------------  进入模型 训练阶段--------------------------------------------------
        self.model.train()
        for iteration, batch_data in enumerate(self.train_data_loader):
            # 数据可能还有其他的标签，根据不同的模型，需要做出不同的修改。这里假设 训练数据集的标签是img和label
            batch_data["img"], batch_data["label"] = batch_data["img"].to(self.device), batch_data["label"].to(self.device)
            
            # ** 固定写法1 ====》 清空梯度
            self.optimizer.zero_grad()  
            if not self.is_fp16:
                pred = self.model(batch_data["img"])
            else:
                with torch.cuda.amp.autocast():
                    # 模型前向传播
                    pred = self.model(batch_data["img"])
                    # 计算损失
            loss = self.loss_fn(pred, batch_data["label"])
            # ** 固定写法2 ====》 反向传播
            loss.backward()
            if self.is_clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            ## ** 固定写法3 ====》 更新参数
            self.optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix(**{'loss': train_loss / (iteration + 1),
                                       'lr': self.optimizer.param_groups[0]['lr'],
                                      })
            train_pbar.update(1)
        # 当一个 epoch训练完成后，关闭进程条
        train_pbar.close()
        print("Training finished!")
        # -------------------------------------------------  进入模型 验证阶段--------------------------------------------------
        # 进入验证模型
        print("Start validating...")
        val_pbar = tqdm(total=len(self.val_data_loader), desc="Validation epoch:{}/{}",  postfix=dict, mininterval=0.3)
        self.model.eval()
        
        
        # 判断是否需要保存模型
        for iteration, batch_data in enumerate(self.val_data_loader):
            # 数据可能还有其他的标签，根据不同的模型，需要做出不同的修改。这里假设 训练数据集的标签是img和label
            batch_data["img"], batch_data["label"] = batch_data["img"].to(self.device), batch_data["label"].to(self.device)
            
            with torch.no_grad():
                if not self.is_fp16:
                    pred = self.model(batch_data["img"])
                else:
                    with torch.cuda.amp.autocast():
                        # 模型前向传播
                        pred = self.model(batch_data["img"])
                        # 计算损失
                loss = self.loss_fn(pred, batch_data["label"])
            val_loss += loss.item()
            val_pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
            })  
            val_pbar.update(1)
        # 当一个 epoch验证完成后，关闭进程条
        val_pbar.close()
        print("Validation finished!")
        # 计算训练和验证的准确率 ---> 判断是否需要保存模型
        cur_train_loss, cur_val_loss = train_loss / len(self.train_data_loader), val_loss / len(self.val_data_loader)
        self.loss_history.append_loss(cur_epoch +1, cur_train_loss, cur_val_loss)
        print(f"total-epoch:{self.epoch_num}, this epoch:{cur_epoch+1}")
        print('Total Loss: %.3f || Val Loss: %.3f ' % (cur_train_loss, cur_val_loss))
        # --------------------------------- 保存权重 --------------------------------
        if len(self.loss_history.val_loss) <= 1 or cur_val_loss <= min(self.loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, "best_epoch_weights.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, f"best_epoch_weights_epoch:{cur_epoch+1}_val_loss:{cur_val_loss}_.pth"))
        return cur_train_loss, cur_val_loss
    
    def train(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for cur_epoch in range(epoch):
            # 训练一个epoch
            train_loss, val_loss = self.fit_one_epoch(cur_epoch)
            # 记录训练过程中的损失和准确率
            total_loss += train_loss
            # 计算准确率
            # correct += (pred.argmax(dim=1) == batch_data["label"]).sum().item()
            # total += batch_data["label"].size(0)
            ...
























