from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
config = UperNetConfig(backbone_config=backbone_config)
model=UperNetForSemanticSegmentation(config)

print(model)

