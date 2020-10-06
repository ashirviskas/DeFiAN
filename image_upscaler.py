import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import imageio
from model import Generator
import argparse
import numpy as np
from torch.autograd import Variable

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])

class ImageUpscaler():

    def __init__(self, modelpath):
        self.model = Generator(64, 20, 10, nn.ReLU(True), True, scale=[2]) # TODO: Make these params dynamic
        self.model.load_state_dict(torch.load(modelpath, map_location = {'cuda:1': 'cuda:0'})["state_dict"], strict=False)
        self.model = self.model.cuda()

    @staticmethod
    def read_image(image_path):
        return imageio.imread(image_path)

    @staticmethod
    def np_to_tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        torch_tensor = torch.from_numpy(np_transpose).float()
        return torch_tensor

    def upscale_image(self, image):
        image = image/255
        image = self.np_to_tensor(image)
        image = normalize(image)
        image = Variable(image).view(1, *image.shape)
        image = image.cuda()

        upscaled = self.model(image).data[0].cpu()
        upscaled = unnormalize(upscaled)
        upscaled = upscaled.mul(255).clamp(0, 255).round()
        upscaled = upscaled.numpy().astype(np.uint8)
        upscaled = upscaled.transpose((1, 2, 0))
        upscaled = Image.fromarray(upscaled)

        return upscaled

    def upscale_images_in_path(self, images_path, result_path='./results'):
        image_filenames = os.listdir(images_path)
        with torch.no_grad():
            for image_filename in image_filenames:
                image = self.read_image(os.path.join(images_path, image_filename))
                upscaled_image = self.upscale_image(image)
                upscaled_image.save(os.path.join(result_path, image_filename))


def main():
    ip = ImageUpscaler('/home/mati/models/DeFiAN/DeFiAN_L_x2.pth')
    ip.upscale_images_in_path('/tmp/test_images', '/tmp/test_results')


if __name__ == '__main__':
    main()
