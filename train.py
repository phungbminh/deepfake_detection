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
    print(image_path)
    # loop through images folder
    for rootpath, dirnames, filenames in os.walk(image_path):
        for dirname in dirnames:
            if dirname == 'fake_deepfake':
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
def main():
    parser = argparse.ArgumentParser(description='CNN Deepfake detection')
    parser.add_argument('--device', type=int, default=0,help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 25)')
    parser.add_argument('--networkInfo', type=str, default='InceptionV3', help='InceptionV3, ResNet18 , or VGG16 (default: InceptionV3)')
    parser.add_argument('--image_path', type=str, default='/kaggle/input/deep-fake/images/')
    args = parser.parse_args()
    print(args)
    clone_data(args.image_path)

    # define img transformers and create dataLoaders
    train_tranform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, expand=False, fill=None),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tranform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = DFD_dataset(train_path, transforms=train_tranform)
    trainloader = dataloader.DataLoader(train_data, batch_size=60, shuffle=True)

    val_data = DFD_dataset(val_path, transforms=val_tranform)
    valloader = dataloader.DataLoader(val_data, batch_size=60, shuffle=True)

    # if cuda is available, use GPU to accelerate training process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # init network (pre-trained on ImageNet)
    network = models.inception_v3(pretrained=True)
    # input channels to fc
    num_fc_in = network.fc.in_features
    # change out features to 2 (fit our binary classification task)
    network.fc = nn.Linear(num_fc_in, 2)
    network = network.to(device)

    # define loss function
    criterion = nn.CrossEntropyLoss()
    # set different learning rate for revised fc layer and previous layers
    # add weight decay (L2 Regularization)
    lr = 0.008 / 10
    fc_params = list(map(id, network.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, network.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': network.fc.parameters(), 'lr': lr * 10}],
        lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    # learning rate decay function
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # training process
    epoch_num = args.epochs
    # data recorders
    training_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    # change accordingly
    model_root_path = current_path + '/model_res/'
    networkInfo = args.networkInfo
    model_path = model_root_path + networkInfo
    # make dirs
    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # force pseudorandom to generate 20 random seeds for reproduce
    random.seed(88)
    seeds = random.sample(range(0, 88), args.epochs)

    # begin training
    for epoch in range(epoch_num):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        network.train()
        running_loss = 0.0
        # set random seed for current epoch
        set_seed(seeds[epoch])
        # Lấy tổng số batch trong trainloader
        total_batches = len(trainloader)

        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, x = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Tính toán phần trăm hoàn thành
            percent_complete = (i + 1) / total_batches * 100
            print( f'\rEpoch [{epoch + 1}/{epoch_num}], Batch [{i + 1}/{total_batches}] - Processing: {percent_complete:.2f}%', end='')

            # calculate loss and accuracy
        print('Evaluating...')
        epoch_loss = running_loss / len(trainloader)
        training_loss.append(epoch_loss)
        train_acc.append(test(network, trainloader, optimizer))
        val_acc.append(test(network, valloader, optimizer))
        # whether to save current model
        save_local(network, model_path, epoch)
        # print result of current epoch
        print({'training_loss': training_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        # step forward the scheduler function
        scheduler.step()

        # end training
        print('Finished Training')





if __name__ == "__main__":
    main()