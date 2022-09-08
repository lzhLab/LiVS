from backbone import resnet

def build_backbone(back_bone):
    if back_bone == "resnet50":
        return resnet.ResNet50(pretrained=False)

