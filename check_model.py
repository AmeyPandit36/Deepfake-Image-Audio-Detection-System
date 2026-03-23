import torch
import json

state_dict = torch.load('models/resnet_image_classifier_v1.pth', map_location='cpu')

fc_shapes = {k: list(v.shape) for k, v in state_dict.items() if k.startswith('fc')}

is_resnet18 = 'layer4.1.conv2.weight' in state_dict and 'layer4.2.conv1.weight' not in state_dict
is_resnet34 = 'layer4.2.conv1.weight' in state_dict and 'layer4.2.conv3.weight' not in state_dict
is_resnet50 = 'layer4.2.conv3.weight' in state_dict

print("ResNet Type:")
if is_resnet18: print("resnet18")
elif is_resnet34: print("resnet34")
elif is_resnet50: print("resnet50")
else: print("Unknown")

print("\nFC Shapes:")
print(json.dumps(fc_shapes, indent=2))
