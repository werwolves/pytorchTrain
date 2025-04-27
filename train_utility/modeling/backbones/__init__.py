__all__ = ["build_backbone"]


def build_backbone(config):
    
    from .rec_mobilenet_v3 import MobileNetV3
    support_backbone = [
        "MobileNetV3"
    ]
    assert config["name"] in support_backbone, f"backbone {config['name']} not supported!"
    module_name = config.pop("name")
    
    backbone = eval(module_name)(**config)
    
    return backbone
    
    
    