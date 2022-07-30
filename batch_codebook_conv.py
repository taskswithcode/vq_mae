import os
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


def file_exists(file_name):
    try:
        fp = open(file_name)
        fp.close()
        return True
    except:
        return False

def batch_conv(results):
    input_dir = results.input
    size = results.size
    augment = results.augment
    custom_dataset = cs.VQDataLoader(root_dir=input_dir,output_base=results.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (augment > 0):
        model = conv.load_model(device)
    if (augment > 1):
        trans = transforms.Compose([
            transforms.RandomResizedCrop(size=(size, size),scale=(0.2,1.0),interpolation=3), # 3 is bicubic
            transforms.RandomHorizontalFlip()
          ])

    else:
        print("Using single image tranform")
        trans = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size)
      ])

    total = len(custom_dataset)
    for i in range(total):
        sample = custom_dataset[i]
        #print(sample["input"],sample["output_dir"])
        file_name = sample["input"].split("/")[-1]
        img = PIL.Image.open(sample["input"]).convert("RGB")
        iter_count = augment if augment >= 1 else 1
        for j in range(iter_count): 
            suffix = ".npy" if augment >= 1 else ".png"
            output_file = sample["output_dir"] + "/" + ".".join(file_name.split(".")[:-1]) + "_" + str(j) + suffix
            if (file_exists(output_file)):
                print("File exists:",output_file," skipping")
                continue
            rescaled_sample = trans(img)
            if (augment == 0):
                rescaled_sample.save(output_file)
            else: 
                x_vqgan = conv.preprocess(rescaled_sample, target_image_size=size)
                z,indices = conv.convert_to_code(x_vqgan,model,device)
                conv.save_vectors(z.detach().cpu(),output_file) 
            print("{:,}".format(i),"{:,}".format(j), " of ","{:,}".format(total),":",output_file)

        
        
        #print(i, sample['image'].shape, sample['landmarks'].shape)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch codebook conv. This just does rescale if input augment is ==0 ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input", required=True,help='Directory of images')
    parser.add_argument('-output', action="store", dest="output", required=True,help='Output Directory of transformed images')
    parser.add_argument('-size', action="store", dest="size", default=384,help='Size of converted image before codebook conv')
    parser.add_argument('-augment', action="store", dest="augment", default=2,type=int,help='Count of augmentations to generate for a single image')
    results = parser.parse_args()

    batch_conv(results)
