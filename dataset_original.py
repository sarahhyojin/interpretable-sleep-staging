from PIL import Image
import PIL
import numpy as np
from multiprocessing import Pool
import os
from tqdm import tqdm
import cv2
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms


def transform(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.float32)
    upper = img[129:473]
    flow = img[473:602]
    flow_resized = cv2.resize(flow, (1920, 43), interpolation=cv2.INTER_AREA)
    breathing = img[602:731]
    # bottom = img[817:903]
    # concat = np.concatenate((upper, flow_resized, breathing, bottom), axis=0)
    # concat_pil = Image.fromarray(np.uint8(concat))
    # concat_pil = concat_pil.resize((224, 224))
    # final = np.array(concat_pil, dtype=np.float32)/255.0
    # final = np.transpose(final, (2, 0, 1))
    oxy = img[903:1075]
    oxy_resized = cv2.resize(oxy, (1920, 86), interpolation=cv2.INTER_AREA)
    concat = np.concatenate((upper, flow_resized, breathing, oxy_resized), axis=0)
    concat_pil = Image.fromarray(np.uint8(concat))
    concat_pil = concat_pil.resize((224, 224))
    final = np.array(concat_pil, dtype=np.float32)/255.0
    final = np.transpose(final, (2, 0, 1))
    return final

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        df = pd.read_csv(annotations_file, sep="\t", header=None)
        self.labels = dict(zip(df[0], df[1]))
        self.image_path = list(self.labels.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # bring the image from corresponding location
        img_path = self.image_path[idx]
        label = int(self.labels[img_path])
        
        # process the image
        image = torch.from_numpy(transform(img_path))

        return image, label
