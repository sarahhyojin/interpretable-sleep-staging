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
from sklearn.metrics import f1_score
import timm

from dataset_original import CustomImageDataset
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def model_test(model, data_loader, device, save_path):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('Wake', 'N1', 'N2', 'N3', 'REM')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1).reshape((5, 1)), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (5,4))
    sn.heatmap(df_cm, annot=True, cmap="Blues")

    # Save confusion matrix plot to the specified directory
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{current_date}.png'))

    # Show confusion matrix plot
    plt.show()


    # classification report
    report = classification_report(y_true, y_pred, target_names = classes, digits=4)
    report_filename = 'classification_report_test.txt'
    report_path = os.path.join(save_path, report_filename) if save_path else report_filename

    print(report)

    with open(report_path, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()


    # large
    parser.add_argument('--test_label_path', type=str, default="/tf/data_AIoT1/psg_image/labels/test_path_1216.txt")
    parser.add_argument('--model_name', type=str, default='ViT-full-v2-')
    parser.add_argument('--checkpoint_path', type=str, default='/tf/data_AIoT1/ViT_models/Full-cropped_v2-2023-12-18.pth')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=360)

    args=parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    # Dataset
    test_dataset = CustomImageDataset(args.test_label_path)

    # Set seed
    torch.manual_seed(args.seed)

    # Data load
    print("========== Load Data ==========")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    
    # Get the model
    num_classes = 5
    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.to(device)

    # Load the model
    model_path = args.checkpoint_path
    model.load_state_dict(torch.load(model_path))


    # Test
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the current date
    model_name = args.model_name + current_date


    print("========= Test Model =========")
    save_path = '/tf/data_AIoT1/psg_image_codes/confusion_matrix/'
    model_test(model, test_dataloader, device, save_path)
  
