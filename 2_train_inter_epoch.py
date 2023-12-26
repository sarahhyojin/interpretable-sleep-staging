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
import timm

from inter_epoch_transformer import InterEpochTransformer
import sys
sys.path.append('/tf/data_AIoT1/psg_image_codes/train_code/')
from dataset_200_v3 import TrainImageDataset
import wandb
import datetime
import pickle

def model_train(model, data_loader, loss_fn, optimizer, device, num_seq):

    model.train()
    running_loss = 0
    corr = 0
    all_preds = []
    all_labels = []
    losses = []
    attention_scores_list = []

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        # print(lbl, type(lbl))
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        outputs = model(img) # 200 outputs: [out_1,out_2,out_3,out_4,out_5]
        # attention_scores_list.append(weights_1.detach().cpu().numpy())
        # attention_scores_list.append(weights_2.detach().cpu().numpy())
        # attention_scores_list.append(weights_3.detach().cpu().numpy())
        # attention_scores_list.append(weights_4.detach().cpu().numpy())
        # print(outputs[0].shape)
        
        # calculate loss by each epoch
        loss = 0
        
        for ep in range(num_seq):
            loss += loss_fn(outputs[ep], lbl[:,ep])
            _, pred = outputs[ep].max(dim=1)
            corr += pred.eq(lbl[:,ep]).sum().item() # number of correct predictions
            # print(pred.cpu())
            # print(lbl[:,ep].cpu())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbl[:,ep].cpu().numpy())
            
        loss.backward()
        optimizer.step()
        losses.append(loss)
    
    # avg_loss = sum(losses) / len(losses)
    avg_loss = sum(losses) / (len(losses) * num_seq)
    acc = corr / (len(data_loader.dataset) * num_seq)
    # print(all_labels, all_preds)
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    # print(f"training loss: {avg_loss:.5f}, acc: {acc:.5f}, f1: {f1: .5f}")
    return all_labels, all_preds, avg_loss, acc, f1


def model_evaluate(model, data_loader, loss_fn, device, num_seq):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        all_preds = []
        all_labels = []
        losses = []
        attention_scores_list = []
        
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
            img, lbl = img.to(device), lbl.to(device)
            
            outputs = model(img) # 200 outputs: [out_1,out_2,out_3,out_4,out_5]
            # attention_scores_list.append(weights_1.detach().cpu().numpy())
            # attention_scores_list.append(weights_2.detach().cpu().numpy())
            # attention_scores_list.append(weights_3.detach().cpu().numpy())
            # attention_scores_list.append(weights_4.detach().cpu().numpy())
            # print(outputs[0].shape)
            
            loss = 0
            # calculate loss by each epoch
            for ep in range(num_seq):
                loss += loss_fn(outputs[ep], lbl[:,ep])
                _, pred = outputs[ep].max(dim=1)
                corr += pred.eq(lbl[:,ep]).sum().item() # number of correct predictions
                # print(pred.cpu())
                # print(lbl[:,ep].cpu())
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl[:,ep].cpu().numpy())
                # running_loss += loss.item() * img.size(0)
                
            losses.append(loss)
            
        # avg_loss = sum(losses) / len(losses)
        avg_loss = sum(losses) / (len(losses) * num_seq)
        acc = corr / (len(data_loader.dataset) * num_seq)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return all_labels, all_preds, avg_loss, acc, f1


if __name__ == '__main__':
    tqdm._instances.clear()
    wandb.init(project="Full")
    # args
    parser = argparse.ArgumentParser()

    # small
    # parser.add_argument('--train_label_path', type=str, default='/tf/hjlee/psg_image/A2020-NX-01/labels/all_0522.txt')
    # parser.add_argument('--eval_label_path', type=str, default='/tf/hjlee/psg_image/A2019-NX-01/labels/test_0525.txt')

    # large
    parser.add_argument('--train_paths', type=str, default="/tf/data_AIoT1/psg_image/train_vector_15_v2/")
    parser.add_argument('--train_labels', type=str, default="/tf/data_AIoT1/psg_image/labels/train_15_label.txt")
    parser.add_argument('--valid_paths', type=str, default="/tf/data_AIoT1/psg_image/valid_vector_15_v2/")
    parser.add_argument('--valid_labels', type=str, default="/tf/data_AIoT1/psg_image/labels/valid_15_label.txt")
    parser.add_argument('--model_name', type=str, default='Seq-Full-15-relprop-final-relu-mlp-')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_seq', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=1e-8)
    args=parser.parse_args()
    wandb.config.update(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    train_dataset = TrainImageDataset(args.train_paths, args.train_labels)
    eval_dataset = TrainImageDataset(args.valid_paths, args.valid_labels)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    # Add seq-attention & Multi-GPU option
    model = InterEpochTransformer(num_classes=5, embed_dim=768, depth=4,
                 num_heads=8, num_seq=15, mlp_ratio=0.5, qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    model = nn.DataParallel(model)
    model.to(device)

    # Optimizer & Loss function
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5) 

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay) # this is the best
    # optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4) # experiment
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
    num_seq = args.num_seq

    min_loss = np.inf

    print("========= Train Model =========")
    for epoch in range(0, num_epochs, 1): # evaluate every epoch
        train_labels, train_preds, train_loss, train_acc, train_f1 = model_train(model, train_dataloader, loss_fn, optimizer, device, num_seq)
        val_labels, val_preds, val_loss, val_acc, val_f1 = model_evaluate(model, eval_dataloader, loss_fn, device, num_seq)

        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), f'/tf/data_AIoT1/ViT_models/{model_name}.pth')
            # with open(f'/tf/data_AIoT1/Att_weights/{model_name}_weights.pkl', 'wb') as w:
            #     pickle.dump(val_weights, w)
            with open(f'/tf/data_AIoT1/Att_weights/{model_name}_labels.pkl', 'wb') as l:
                pickle.dump(val_labels, l)
            with open(f'/tf/data_AIoT1/Att_weights/{model_name}_preds.pkl', 'wb') as p:
                pickle.dump(val_preds, p)
  
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, f1: {train_f1:.5f},\
                val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}, val_f1: {val_f1:.5f}')

        wandb.log({
        "Train Accuracy": 100. * train_acc,
        "Train Loss": train_loss,
        "Train F1": 100 * train_f1,
        "Test Accuracy": 100. * val_acc,
        "Test Loss": val_loss,
        "Test F1": 100 * val_f1})
