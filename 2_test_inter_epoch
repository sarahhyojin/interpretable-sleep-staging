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

import sys
sys.path.append('/tf/data_AIoT1/psg_image_codes/train_code/')
from inter_epoch_transformer import InterEpochTransformer
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import OrderedDict

def sliding_windows(patient_vector, num_seq, batch_size, length, model, device):
    """
    patient_vector : extracted feature vectors per image and concatenated from one patient
    num_seq : sliding window's kernel size
    batch_size : batch size to be processed
    length : length of the total epoch of one patient
    model : trained model
    final : return the prediction of sequence that of one patient (softmax aggregated)
    """
    final = [torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32, device=device) for _ in range(length)]

    current_seq = []

    with torch.no_grad():
        for i in range(length - num_seq + 1):
            current_seq.append(patient_vector[i:i+num_seq])

            if (i+1) % batch_size == 0:
                batch_seq = torch.stack(current_seq).to(device)
                current_seq = []
                output = model(batch_seq)

                for k in range(batch_size):  # batch
                    for j in range(num_seq):
                        final[k + j + i - (batch_size-1)].add_(torch.softmax(output[j][k], dim=0))

        # for remainder
        if len(current_seq) != 0:
            batch_seq = torch.stack(current_seq).to(device)
            output_2 = model(batch_seq)

            new_i = batch_size * ((length - num_seq + 1)//batch_size)

            for k in range(output_2[0].shape[0]):  # batch
                for j in range(num_seq):  # sequence length
                    final[k + j + new_i].add_(torch.softmax(output_2[j][k], dim=0))

    return final

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()


    # large
    parser.add_argument('--img_dir', type=str, default="/tf/data_AIoT1/psg_image")
    parser.add_argument('--test_labels', type=str, default="/tf/data_AIoT1/psg_image/labels/test_1209.txt") # EM is modified
    parser.add_argument('--model_name', type=str, default='Seq-sliding-test-32')
    parser.add_argument('--checkpoint_path', type=str, default="/tf/data_AIoT1/ViT_models/Seq-Full-15-relprop-final-relu-2023-12-18.pth") # use one with relprop
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512) # in cpu
    parser.add_argument('--num_seq', type=int, default=15)
    parser.add_argument('--save_path', type=str, default = "/tf/data_AIoT1/psg_image_codes/confusion_matrix/")

    args=parser.parse_args()

     # Set seed
    torch.manual_seed(args.seed)

    # read test_labels
    df_file_list = pd.read_csv(args.test_labels, sep="\t", header=None)
    files = df_file_list[0].tolist()

    # create patient_dict
    patient_dict = {} # number of vectors that is from the same patient
    prev_patient = None

    for file in files:
        patient = file[0:16]
        if patient == prev_patient:
            patient_dict[patient] += 1
        else:
            patient_dict[patient] = 1
        prev_patient = patient

    
    y_softmax = []
    y_pred = []
    y_true = []

    # model
    print("========= Load Model =========")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    model = InterEpochTransformer(num_classes=5, embed_dim=768, depth=4,
        num_heads=8, num_seq=15, mlp_ratio=0.5, qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    model = nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint)
    model.to(device)

    print("========= Process by patient =========")
    for patient, length in tqdm(patient_dict.items(), total=len(patient_dict)):
        hospital = '-'.join(patient.split('-')[:-1])
        patient_folder = os.path.join(args.img_dir, hospital, "vectors_v2", patient)
        if not os.path.exists:
            print(patient_folder)
        temp = []
        lbl = []
        patient_epoch = df_file_list[df_file_list[0].str.startswith(patient)]
        sorted_epoch = patient_epoch.sort_values(by=0)
        # sorted_epoch = sorted(os.listdir(patient_folder))
        # print(sorted_epoch)
        for index, row in sorted_epoch.iterrows():
            # print(f"Loading {i}th vector")
            vector_np = torch.from_numpy(np.load(os.path.join(patient_folder, row[0])).astype(np.float32))
            temp.append(vector_np)
            lbl.append(row[1])
            # print(len(temp))
        patient_vector = torch.stack(temp)

        final = sliding_windows(patient_vector=patient_vector, num_seq=args.num_seq, batch_size = args.batch_size,
                                length = length, model = model, device=device)
        y_softmax.extend(final)
        y_true.extend(lbl)
        
        # print(patient_vector.shape, len(lbl))
    
    print("========= Inference Ended =========")

    for seq in y_softmax:
        _, pred = seq.max(dim=0)
        y_pred.append(pred.item())

    classes = ('Wake', 'N1', 'N2', 'N3', 'REM')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1).reshape((5, 1)), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (5,4))
    sn.heatmap(df_cm, annot=True, cmap="Blues")

    # Save confusion matrix plot to the specified directory
    plt.savefig(os.path.join(args.save_path, f'confusion_matrix_{args.model_name}.png'))

    # Show confusion matrix plot
    plt.show()

    # classification report
    report = classification_report(y_true, y_pred, target_names = classes, digits=4)
    print(report)
    report_filename = f'classification_report_test_{args.model_name}.txt'
    report_path = os.path.join(args.save_path, report_filename)

    with open(report_path, 'w') as f:
        f.write(report)
