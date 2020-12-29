from captum.attr._core.deep_lift import DeepLift
import numpy as np
import captum
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import numpy as np
from captum.attr._utils import visualization as viz

model = torchvision.models.vgg19(pretrained=True)
model.eval()

transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 transforms.Resize((224,224))])
im = Image.open('elephant.jpeg')
im = transform(im)
im = im.unsqueeze_(0)
print("imshap", im.shape)
im.requires_grad = True
target =  torch.argmax(model(im))
print("target is ", target)
baseline = torch.zeros((1,3,224,224,))
baseline.requires_grad = True

ig = DeepLift(model)
print(im.shape, baseline.shape)
attributions= ig.attribute(im, baseline, target=target)
print('IG Attributions:', attributions)
print("attrib shape, im shape", attributions.shape, im.shape)
'''
at_v, im_v = attributions.detach().numpy().squeeze().swapaxes(0,2), im.detach().numpy().squeeze().swapaxes(0,2)
mer = captum.attr.visualization.visualize_image_attr(at_v, im_v, "heat_map")
torchvision.utils.save_image(mer, "mer_out.png")
'''
attributions = torch.where(attributions>0, attributions, torch.zeros(attributions.shape))
print(attributions.min())
print(attributions.max())
attributions /= attributions.max()
torchvision.utils.save_image(attributions, "out.png")
