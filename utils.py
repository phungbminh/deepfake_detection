# necessary dependencies
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import argparse
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import random


import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import dataloader


# define training and validation dataset class
class DFD_dataset(Dataset):
    def __init__(self, img_path, transforms=None):
        self.transforms = transforms
        self.img_path = img_path
        self.real_dir = img_path + 'real'
        self.fake_dir = img_path + 'fake'
        self.real_num = len(os.listdir(self.real_dir))
        self.fake_num = len(os.listdir(self.fake_dir))

    def __len__(self):
        return self.real_num + self.fake_num

    def __getitem__(self, index):
        if index < self.real_num:
            label = 1
            img = Image.open(self.real_dir + '/' + str(index) + '.png')
        else:
            label = 0
            img = Image.open(self.fake_dir + '/' + str(index - self.real_num) + '.png')

        if self.transforms:
            img = self.transforms(img)

        return img, label

def train(network, trainloader, optimizer, criterion, device, seeds, epoch, networkInfo):
    network.train()
    running_loss = 0.0
    set_seed(seeds[epoch])  # Set random seed for the current epoch
    total_batches = len(trainloader)

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        if networkInfo == 'ResNet50' or networkInfo == 'VGG19':
            outputs = network(inputs)  # Forward pass
        elif networkInfo == 'InceptionV3':
            outputs, x = network(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()
        percent_complete = (i + 1) / total_batches * 100
        print(f'\rEpoch [{epoch + 1}], Batch [{i + 1}/{total_batches}] - Processing: {percent_complete:.2f}%', end='')

    epoch_loss = running_loss / len(trainloader)
    return epoch_loss

# define test function to calculate both training and val accuracy
def test(network, loader, optimizer, device):
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return round((float(100) * float(correct) / float(total)), 4)

# define this function to save state_dict of each epoch
def save_local(best_model, rootpath, name):
    path = rootpath + '/' + name + '.pt'
    torch.save(best_model, path)

# define function to set random seed for each epoch
def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate(network, dataloader, optimizer):
    accuracy = test(network, dataloader, optimizer)
    return accuracy

def predict_image(image_path, networkInfo):
    if networkInfo == 'ResNet50':
        img_size = 224
        network = models.resnet50(pretrained=True)
    elif networkInfo == 'VGG19':
        img_size = 224
        network = models.vgg19(pretrained=True)
    elif networkInfo == 'InceptionV3':
        img_size = 299
        network = models.inception_v3(pretrained=True)
    else:
        raise ValueError("Unsupported network: choose from InceptionV3, ResNet50, or VGG19")
    test_tranform = transforms.Compose([
        # transforms.Resize(299),
        # transforms.CenterCrop(299),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model_root_path = os.getcwd() + '/model_res/'
    model_path = os.path.join(model_root_path, networkInfo)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_fc_in = network.fc.in_features
    network.fc = nn.Linear(num_fc_in, 2)
    network.load_state_dict(torch.load( model_path + '/BestTrain.pt', map_location=device))
    network.to(device)
    network.eval()

    img = Image.open(image_path).convert("RGB")
    img_tensor = test_tranform(img).unsqueeze(0).to(device)  # Thêm chiều batch


    with torch.no_grad():
        outputs = network(img_tensor)
        _, predicted = torch.max(outputs, 1)

    result = 'Real' if predicted.item() == 1 else 'Fake'

    plt.imshow(img)
    plt.title(f'Prediction: {result}')
    plt.axis('off')
    plt.show()