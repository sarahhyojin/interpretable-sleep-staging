import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, CenterCrop
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score

from dataset_original import CustomImageDataset
import wandb
import datetime
import timm

def model_train(model, data_loader, loss_fn, optimizer, device):

    model.train()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        _, pred = output.max(dim=1)
        
        # accuracy
        corr += pred.eq(lbl).sum().item()
        # macro f1 score
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(data_loader.dataset), acc, f1


def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        all_preds = []
        all_labels = []
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
            img, lbl = img.to(device), lbl.to(device)

            output = model(img)
            _, pred = output.max(dim=1)

            corr += torch.sum(pred.eq(lbl)).item()
            # macro f1 score
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            running_loss += loss_fn(output, lbl).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return running_loss / len(data_loader.dataset), acc, f1

if __name__ == '__main__':
    tqdm._instances.clear()
    wandb.init(project="Full")
    # args
    parser = argparse.ArgumentParser()

    # large
    parser.add_argument('--train_label_path', type=str, default='/tf/data_AIoT1/psg_image/labels/train_path_1216.txt')
    parser.add_argument('--eval_label_path', type=str, default='/tf/data_AIoT1/psg_image/labels/valid_path_1216.txt')
    parser.add_argument('--model_name', type=str, default='Full-cropped_v2-')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=360)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=1e-4)

    args=parser.parse_args()
    wandb.config.update(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    train_dataset = CustomImageDataset(args.train_label_path)
    eval_dataset = CustomImageDataset(args.eval_label_path)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    
    # Get the model
    num_classes = 5
    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.to(device)

    # Load the model
    # model_path = "/tf/hjlee/ViT_models/ViT-2023-05-25.pth"
    # model.load_state_dict(torch.load(model_path))

    # Optimizer & Loss function
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay) # this is best
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    wandb.watch(model)
    loss_fn = nn.CrossEntropyLoss()

    # Create a cosine annealing learning rate scheduler
    # steps_per_epoch = len(train_dataset) // args.batch_size
    # scheduler = lr_scheduler.stepLR(optimizer, T_max=args.epochs * steps_per_epoch)

    # Train
    num_epochs = args.epochs
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the current date
    model_name = args.model_name + current_date

    min_loss = np.inf

    print("========= Train Model =========")
    for epoch in range(0, num_epochs, 1): # evaluate every epoch
        train_loss, train_acc, train_f1 = model_train(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc, val_f1 = model_evaluate(model, eval_dataloader, loss_fn, device)

        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), f'/tf/data_AIoT1/ViT_models/{model_name}.pth')
  
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, f1: {train_f1:.5f},\
                val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}, val_f1: {val_f1:.5f}')

        wandb.log({
        "Train Accuracy": 100. * train_acc,
        "Train Loss": train_loss,
        "Train F1": 100 * train_f1,
        "Test Accuracy": 100. * val_acc,
        "Test Loss": val_loss,
        "Test F1": 100 * val_f1})
