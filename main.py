from __future__ import print_function
import torch
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, required=True)
parser.add_argument('--k_size', type=int, default=15)
args = parser.parse_args()

assert os.path.exists(args.input_image)
assert args.k_size >= 3 and args.k_size % 2 == 1

# check cuda support
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

print('Loading pretrained model')
model = torchvision.models.vgg16(pretrained=True)
model.eval()
if HAS_CUDA:
    model.cuda()

print('Load and [pre]process image')
image_orig = Image.open(args.input_image)
image_orig = image_orig.resize((224, 224), Image.NEAREST)
image_orig = np.asarray(image_orig)

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

image = transf(image_orig).unsqueeze(0)
if HAS_CUDA:
    image = image.cuda()

pred = model(torch.autograd.Variable(image))
pred = pred.data.cpu().numpy().squeeze()

class_id, class_label, class_prob = utils.predictions_to_class_info(pred)
print(class_id, class_label, class_prob)

print('Computing saliency map')
smap = utils.compute_saliency_map(model, image[0], args.k_size, class_id, class_prob)

from scipy.misc import imresize
mask = imresize(smap, image_orig.shape[:2])

plt.figure(1)
plt.subplot(1,3,1)
plt.title(class_label)
plt.imshow(image_orig)
plt.subplot(1,3,2)
plt.title("Saliency map")
plt.imshow(smap, cmap=plt.get_cmap('jet'))
plt.subplot(1,3,3)
plt.imshow(image_orig)
plt.imshow(mask, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.show()
