import torch
from torch.optim import optimizer
from torchvision import models
from model import model 
from torch import optim
from torch import nn 
import torch.nn.functional as F 
import time 
from dataloader import train_data_loader, val_data_loader
from model import device
import os 
from tqdm import tqdm
import numpy as np 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

optimizer = optim.Adam(model.parameters(), lr= 0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
criterion = nn.CrossEntropyLoss()
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
epochs = 5

def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)

for epoch in tqdm(range(epochs)):
    start = time.time()
    train_epoch_loss = []
    train_epoch_accuracy = []
    _iter = 1

    val_epoch_loss = []
    val_epoch_accuracy = []
    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        acc = calc_accuracy(labels.cpu(), output.cpu())
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        train_epoch_loss.append(loss_value)
        train_epoch_accuracy.append(acc)
        
        if _iter % 500 ==0:
            print("> Iteration {} < ".format(_iter))
            print("Iter Loss = {} ".format(round(loss_value,4)))
            print("Iter Accuracy: {} % \n".format(acc))
        _iter += 1
    
    for images, labels in val_data_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        # Calculate Accuracy
        acc = calc_accuracy(labels.cpu(), output.cpu())
        loss = criterion(output, labels)
        loss_value = loss.item()
        val_epoch_loss.append(loss_value)
        val_epoch_accuracy.append(acc)
    train_epoch_loss = np.mean(train_epoch_loss)
    train_epoch_accuracy = np.mean(train_epoch_accuracy)

    val_epoch_loss = np.mean(val_epoch_loss)
    val_epoch_accuracy = np.mean(val_epoch_accuracy)
    end = time.time()
    
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)

    print("** Epoch {} ** - Epoch Time {}".format(epoch, int(end-start)))
    print("Train Loss = {}".format(round(train_epoch_loss, 4)))
    print("Train Accuracy = {} % \n".format(train_epoch_accuracy))
    print("Val Loss = {}".format((val_epoch_loss, 4)))
    print("Val Accuracy = {} % \n".format(val_epoch_accuracy))
