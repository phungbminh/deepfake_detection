import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
from torch.utils.data import dataloader
import utils


def run(networkInfo, train_path, val_path, device, epochs):
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

    train_data = utils.DFD_dataset(train_path, transforms=train_transform)
    trainloader = dataloader.DataLoader(train_data, batch_size=60, shuffle=True)

    val_data = utils.DFD_dataset(val_path, transforms=val_transform)
    valloader = dataloader.DataLoader(val_data, batch_size=60, shuffle=True)

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo mạng
    if networkInfo == 'ResNet50':
        network = models.resnet50(pretrained=True)
        num_fc_in = network.fc.in_features
        network.fc = nn.Linear(num_fc_in, 2)
    elif networkInfo == 'VGG19':
        network = models.vgg19(pretrained=True)
        num_fc_in = network.classifier[6].in_features
        network.classifier[6] = nn.Linear(num_fc_in, 2)
    elif networkInfo == 'InceptionV3':
        network = models.inception_v3(pretrained=True)
        num_fc_in = network.fc.in_features
        network.fc = nn.Linear(num_fc_in, 2)
    else:
        raise ValueError("Unsupported network: choose from InceptionV3, ResNet50, or VGG19")

    network = network.to(device)

    criterion = nn.CrossEntropyLoss()

    # Đặt tốc độ học và bộ tối ưu hóa với weight decay (L2 Regularization)
    lr = 0.008 / 10

    if networkInfo == 'VGG19':
        fc_params = list(map(id, network.classifier.parameters()))
    else:
        fc_params = list(map(id, network.fc.parameters()))

    base_params = filter(lambda p: id(p) not in fc_params, network.parameters())

    optimizer = optim.Adam([
        {'params': base_params},
        {'params': network.classifier.parameters() if networkInfo == 'VGG16' else network.fc.parameters(),
         'lr': lr * 10}],
        lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epoch_num = epochs
    perf_loss = []
    perf_train_acc = []
    perf_val_acc = []
    test_acc = []

    model_root_path = os.getcwd() + '/model_res/'
    model_path = os.path.join(model_root_path, networkInfo)
    os.makedirs(model_root_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    best_val_acc = 0.0
    best_model_weights = None

    random.seed(88)
    seeds = random.sample(range(0, 88), epoch_num)

    for epoch in range(epoch_num):
        print("=====Epoch {}".format(epoch + 1))
        print('Training...')
        loss = utils.train(network, trainloader, optimizer, criterion, device, seeds, epoch, networkInfo)
        perf_loss.append(loss)
        print('..')
        print('Evaluating...')
        train_acc = utils.test(network, trainloader, optimizer, device)
        perf_train_acc.append(train_acc)

        val_acc = utils.test(network, valloader, optimizer, device)
        perf_val_acc.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = network.state_dict()

        print({'training_loss': loss, 'train_acc': train_acc, 'val_acc': val_acc})

        scheduler.step()
    # Lưu mô hình
    utils.save_local(best_model_weights, model_path, 'BestTrain')
    print('Finished Training')
    return perf_train_acc, perf_val_acc, perf_loss
