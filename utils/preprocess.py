from torchvision import datasets, transforms
import torch
import numpy as np

from .data_constants import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

def normal_transform(image):
    '''
    Normalizes according to IMAGENET data + Converts image to tensor
    '''
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_DEFAULT_MEAN,
                            std = IMAGENET_DEFAULT_STD),
        transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))
    ])
    return transform_img(image)