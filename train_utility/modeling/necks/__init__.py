

__all__ = ["build_neck"]

def build_neck(config):
    """
    Build the neck module.
    Args:
        config (dict): The configuration dictionary for the neck module.
    Returns:
        nn.Module: The constructed neck module.
    """
    from .yolo_neck import YOLONeck
    support_neck = [
        "YOLONeck"
    ]
    assert config["name"] in support_neck, f"neck {config['name']} not supported!"
    module_name = config.pop("name")
    
    neck = eval(module_name)(**config)
    
    return neck