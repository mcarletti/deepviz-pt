from __future__ import print_function
import torch
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, required=True)
args = parser.parse_args()

assert os.path.exists(args.input_image)

# check cuda support
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

print('Loading pretrained model')
model = torchvision.models.vgg16_bn(pretrained=True)
model.eval()
if HAS_CUDA:
    model.cuda()

print('Load image')
image_orig = Image.open(args.input_image)
image_orig = image_orig.resize((224, 224), Image.NEAREST)
image_orig = np.asarray(image_orig)

print('Image preprocessing')
mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

image = transf(image_orig).unsqueeze(0)
if HAS_CUDA:
    image = image.cuda()

print('Computing prediction')
pred = model(torch.autograd.Variable(image))
pred = pred.data.cpu().numpy().squeeze()


# show results
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


pred = softmax(pred)
class_id = np.argmax(pred)
class_prob = pred[class_id]
class_labels = np.loadtxt(open('data/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')
print(class_labels[class_id], pred[class_id])

print('Computing saliency map')
saliency_map = np.zeros((224, 224), dtype=np.float32)
rows, cols, ch = image_orig.shape
k_size = 15
h_size = int(k_size / 2)

for u in range(h_size, rows, h_size):
    for v in range(h_size, cols, h_size):
        masked_image = image[0].clone()
        masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size] = 0  # mask color
        img = masked_image.unsqueeze(0)
        pred = model(torch.autograd.Variable(img)).cpu().data[0][0]
        pred_err = class_prob - pred
        saliency_map[u, v] = pred_err


from scipy.ndimage.filters import gaussian_filter


smap_smoothed = gaussian_filter(saliency_map, h_size)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


plt.figure(1)
plt.subplot(1,3,1)
plt.title(class_labels[class_id])
plt.imshow(image_orig)
plt.subplot(1,3,2)
plt.title("Saliency map")
plt.imshow(smap_smoothed)
plt.subplot(1,3,3)
t = smap_smoothed - np.min(smap_smoothed)
t = smap_smoothed / np.max(smap_smoothed)
blend = 0.5
plt.imshow(rgb2gray(image_orig), alpha=(1 - blend))
plt.imshow((255 * t).astype(np.uint8), alpha=blend)
plt.show()
