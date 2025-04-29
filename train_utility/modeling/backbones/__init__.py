__all__ = ["build_backbone"]


def build_backbone(config):
    
    from .rec_mobilenet_v3 import MobileNetV3
    from .resnet import ResNet
    from .resnet18 import ResNet as ResNet18
    
    support_backbone = [
        "MobileNetV3", "ResNet", "ResNet18"
    ]
    assert config["name"] in support_backbone, f"backbone {config['name']} not supported!"
    module_name = config.pop("name")
    
    backbone = eval(module_name)(**config)
    
    return backbone
    
    
    