# necessary dependencies
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import dataloader

from torch.utils.tensorboard import SummaryWriter


current_path = os.getcwd()
#current_path = 'D:/deepfake-detection'
train_path = current_path + '/train/'
val_path = current_path + '/val/'
train_fake_path = train_path + 'fake/'
train_real_path = train_path + 'real/'
val_fake_path = val_path + 'fake/'
val_real_path = val_path + 'real/'


# need to keep this .ipynb file in the same directory as the images folder
# divede the images provided into training and validation set (8:2)
# create directories
def clone_data(image_path):
    print('Splitting dataset...')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(train_fake_path):
        os.makedirs(train_fake_path)
    if not os.path.exists(train_real_path):
        os.makedirs(train_real_path)
    if not os.path.exists(val_fake_path):
        os.makedirs(val_fake_path)
    if not os.path.exists(val_real_path):
        os.makedirs(val_real_path)

    # distribute 12000 images into different folders
    train_fake_num = 0
    train_real_num = 0

    val_fake_num = 0
    val_real_num = 0

    test_fake_num = 0
    test_real_num = 0

    # loop through images folder
    for rootpath, dirnames, filenames in os.walk(image_path):

        for dirname in dirnames:
            if dirname == 'fake_deepfake':
                print(' > fake_deepfake:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4487)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in fake_deepfake folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_fake_num) + '.png'
                        dstpath = val_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_fake_num += 1
                    else:
                        newname = str(train_fake_num) + '.png'
                        dstpath = train_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_fake_num += 1
                print('done')
            elif dirname == 'fake_face2face':
                print(' > fake_face2face:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4486)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in fake_face2face folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_fake_num) + '.png'
                        dstpath = val_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_fake_num += 1
                    else:
                        newname = str(train_fake_num) + '.png'
                        dstpath = train_fake_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_fake_num += 1
                print('done')
            elif dirname == 'real':
                print(' > real:')
                # generate 800 random number in the range [0, 3999] to represent those go to val
                # force pseudorandom split
                random.seed(4485)
                val_index = random.sample(range(0, 4000), 800)
                # directory full path
                image_folder = rootpath + dirname + '/'
                # loop all images in real folder
                imgfiles = os.listdir(image_folder)
                for imgfile in imgfiles:
                    srcpath = image_folder + imgfile
                    index = int(imgfile.split('.')[0])
                    if index in val_index:
                        newname = str(val_real_num) + '.png'
                        dstpath = val_real_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        val_real_num += 1
                    else:
                        newname = str(train_real_num) + '.png'
                        dstpath = train_real_path + newname
                        shutil.copyfile(srcpath, dstpath)
                        train_real_num += 1
                print('done')


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
        if networkInfo == 'ResNet50' or networkInfo == 'VGG16':
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
def save_local(network, rootpath, epoch):
    path = rootpath + '/' + str(epoch) + '.pt'
    torch.save(network.state_dict(), path)

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

def main():
    parser = argparse.ArgumentParser(description='CNN Deepfake detection')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--networkInfo', type=str, default='InceptionV3', help='InceptionV3, ResNet50, or VGG16 (default: InceptionV3)')
    parser.add_argument('--image_path', type=str, default='/kaggle/input/deep-fake/images/')
    args = parser.parse_args()
    print(args)


    if not os.path.exists(train_path) or not os.path.exists(val_path):
        clone_data(args.image_path)
        #clone_data('D:/deepfake-detection/images/')
    else:
        print('Path {} exist'.format(train_path))

    networkInfo = args.networkInfo
    if networkInfo == 'ResNet50' or networkInfo == 'VGG16':
        img_size = 224
    elif networkInfo == 'InceptionV3':
        img_size = 299
    else:
        raise ValueError("Unsupported network: choose from InceptionV3, ResNet50, or VGG16")

    normalize = [0.485, 0.456, 0.406]

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, expand=False, fill=None),
        transforms.ToTensor(),
        transforms.Normalize(normalize, [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize, [0.229, 0.224, 0.225])
    ])

    train_data = DFD_dataset(train_path, transforms=train_transform)
    trainloader = dataloader.DataLoader(train_data, batch_size=60, shuffle=True)

    val_data = DFD_dataset(val_path, transforms=val_transform)
    valloader = dataloader.DataLoader(val_data, batch_size=60, shuffle=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo mạng
    if networkInfo == 'ResNet50':
        network = models.resnet50(pretrained=True)
        num_fc_in = network.fc.in_features
        network.fc = nn.Linear(num_fc_in, 2)
    elif networkInfo == 'VGG16':
        network = models.vgg16(pretrained=True)
        num_fc_in = network.classifier[6].in_features
        network.classifier[6] = nn.Linear(num_fc_in, 2)
    elif networkInfo == 'InceptionV3':
        network = models.inception_v3(pretrained=True)
        num_fc_in = network.fc.in_features
        network.fc = nn.Linear(num_fc_in, 2)
    else:
        raise ValueError("Unsupported network: choose from InceptionV3, ResNet50, or VGG16")

    network = network.to(device)

    # Định nghĩa hàm mất mát
    criterion = nn.CrossEntropyLoss()

    # Đặt tốc độ học và bộ tối ưu hóa với weight decay (L2 Regularization)
    lr = 0.008 / 10

    if networkInfo == 'VGG16':
        fc_params = list(map(id, network.classifier.parameters()))
    else:
        fc_params = list(map(id, network.fc.parameters()))

    base_params = filter(lambda p: id(p) not in fc_params, network.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': network.classifier.parameters() if networkInfo == 'VGG16' else network.fc.parameters(), 'lr': lr * 10}],
        lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epoch_num = args.epochs
    perf_loss = []
    perf_train_acc = []
    perf_val_acc = []
    test_acc = []

    model_root_path = current_path + '/model_res/'
    model_path = os.path.join(model_root_path, networkInfo)
    os.makedirs(model_root_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    random.seed(88)
    seeds = random.sample(range(0, 88), epoch_num)

    for epoch in range(epoch_num):
        print("=====Epoch {}".format(epoch + 1))
        print('Training...')
        loss = train(network, trainloader, optimizer, criterion, device, seeds, epoch, args.networkInfo)
        perf_loss.append(loss)
        print('..')
        print('Evaluating...')
        train_acc = test(network, trainloader, optimizer, device)
        perf_train_acc.append(train_acc)

        val_acc = test(network, valloader, optimizer, device)
        perf_val_acc.append(val_acc)

        # Lưu mô hình
        save_local(network, model_path, epoch)

        print({'training_loss': loss, 'train_acc': train_acc, 'val_acc': val_acc})

        scheduler.step()

    print('Finished Training')


if __name__ == "__main__":
    main()