import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 加载预训练的 ResNet50 模型
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 提取 ResNet 的前 7 层
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.out_channels= model.layer3[-1].conv3.out_channels
        # 冻结所有参数
        self.freeze_layers()
    def freeze_parameters(self):
        # 遍历所有参数并冻结
        for param in self.parameters():
            param.requires_grad = False 
    
    ##############################
    def freeze_layers(self):
        # 冻结 conv1, bn1, relu, maxpool 和 layer1
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.relu.parameters():
            param.requires_grad = False
        for param in self.maxpool.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
    ##############################
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # x.shape = torch.Size([1, 1024, 14, 14])

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = ResNet()
    out = model(x)
    print(out.shape)  # 输出形状
    print(model.out_channels)