import os,sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
from PIL import Image
import argparse
import customdataloader as cs
import codebook_conv as conv
import PIL
import torchvision.transforms as transforms
import time

PRETRAIN = 1
FINETUNE_TRAIN = 2
FINETUNE_VALIDATE = 3


def loc_convert(device,model,file_name,output_file,call_type):
    size = 384
    if (call_type == PRETRAIN or call_type == FINETUNE_TRAIN):
        #print("Pretrain and Finetune train:Transforming random resize and flip")
        trans = transforms.Compose([
            transforms.RandomResizedCrop(size=(size, size),scale=(0.2,1.0),interpolation=3), # 3 is bicubic
            transforms.RandomHorizontalFlip()
          ])
    else:
        #print("Resize only transform: fine tune eval")
        trans = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size)
            ])
        #trans = transforms.Compose([
        #    transforms.Resize(size=(size, size))
        #  ])
    img = PIL.Image.open(file_name).convert("RGB")
    if (trans is not None):
        rescaled_sample = trans(img)
    else:
        rescaled_sample = img
    img = conv.preprocess(rescaled_sample, target_image_size=384)
    z,indices = conv.convert_to_code(img,model,device)
    #print(output_file)
    conv.save_vectors(z.detach().cpu(),output_file) 

def load_model():
    device = torch.device("cuda:1")
    model = conv.load_model(device)
    return device,model

if __name__ == '__main__':
    device,model = load_model()
    while (True):
        print("Test converting")
        loc_convert(device,model,sys.argv[1],sys.argv[2],int(sys.argv[3]))
        break
        
