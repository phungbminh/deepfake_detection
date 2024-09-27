# necessary dependencies
import os
import random
import argparse

import torch
import torch.nn as nn
from numpy.f2py.auxfuncs import throw_error
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
from torch.utils.data import dataloader
import dataset
import train


#current_path = 'D:/deepfake-detection'


def main():
    parser = argparse.ArgumentParser(description='CNN Deepfake detection')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--networkInfo', type=str, default='InceptionV3', help='InceptionV3, ResNet50, or VGG19 (default: InceptionV3)')
    parser.add_argument('--image_path', type=str, default='/kaggle/input/deep-fake/images/')
    args = parser.parse_args()
    print(args)

    train_path, val_path = dataset.clone_data('D:/deepfake-detection/images/')

    if os.path.exists(train_path) or os.path.exists(train_path):
        train.run(args.networkInfo, train_path, val_path, args.device, args.epochs)
    else:
        throw_error("No data train")





if __name__ == "__main__":
    main()