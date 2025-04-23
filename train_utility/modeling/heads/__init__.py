__all__ = ["build_head"]
def build_head(config):
    from.yolo_head import YOLOHead
    support_head = [
        "YOLOHead"
    ]   
    assert config["name"] in support_head, f"head {config['name']} not supported!"
    module_name = config.pop("name")
    return eval(module_name)(**config)