from .simple_cnn import SimpleCNN

def build_model(model_name: str, in_ch: int, num_classes: int):
    if model_name == "simple-cnn":
        return SimpleCNN(in_ch=in_ch, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
