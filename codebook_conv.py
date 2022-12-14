import sys
import yaml
import torch
from omegaconf import OmegaConf
import pdb
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse
from einops import rearrange
import my_utils as my

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from IPython.display import display, display_markdown

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def load_model(device):
    sys.path.append(".")

    # also disable grad to save memory
    torch.set_grad_enabled(False)




    configvqf4noattn = my.load_config("models/first_stage_models/vq-f4-noattn/config.yaml", display=False)
    model = my.load_vqgan(configvqf4noattn, ckpt_path="models/first_stage_models/vq-f4-noattn/model.ckpt", is_gumbel=False).to(device)

    return model


def save_vectors(vecs,out_file):
    with open(out_file,"wb") as fp:
        np.save(fp,np.array(vecs))

def codebook_examine (params):
    output_file = params.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    log_fp = open(output_file,"w")
    vecs = model.quantize.embedding.weight.tolist()
    save_vectors(vecs,output_file)
    with open("index.txt","w") as fp:
        for i in range(len(vecs)):
            fp.write(str(i+1) + "\n")

def recons_using_codebook_values(params):
    output_dir = params.output
    try:
        os.mkdir(output_dir)
    except:
        print("Output directory:", output_dir," already exists") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    vecs = model.quantize.embedding.weight.tolist()
    codebook_size = len(vecs)
    for i in range(codebook_size):
        z = model.quantize.get_codebook_entry(torch.LongTensor([i]*96*96),None)
        z = z.permute(1,0).contiguous().view(1,3,96,96)
        xrec = model.decode(z)
        xrec = custom_to_pil(xrec[0])
        w, h = xrec.size[0],xrec.size[0]
        img = Image.new("RGB", (w, h))
        img.paste(xrec, (0,0))
        output_file = f"{output_dir}/{i+1}.png"
        img.save(output_file)

def preprocess(img, target_image_size=384):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x


def gen_image_using_image_custom_codes(params):
    input_file = params.input
    output_file = "regen_" +  ''.join(input_file.split(".")[:-1]) +  ".png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    arr = []
    index_correct = 1
    with open(input_file) as fp:
        for line in fp:
            line = int(line.rstrip("\n")) - index_correct 
            arr.append(line)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = torch.LongTensor(arr)
    arr = arr.to(device)
    z = model.quantize.get_codebook_entry(arr,None)
    z = z.permute(1,0).contiguous().view(1,3,96,96)
    xrec = model.decode(z)
    xrec = custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    img.save(output_file)

def gen_image_using_image_custom_codes2(params):
    input_file = params.input
    output_file = "regen_" +  ''.join(input_file.split(".")[:-1]) +  ".png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    arr = np.load(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.from_numpy(arr).to(device)
    xrec = model.decode(z)
    xrec = custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    img.save(output_file)
            
def convert_to_code(x_vqgan,model,device):
    x_vqgan = x_vqgan.to(device)
    x = preprocess_vqgan(x_vqgan)
    z, _, [_, _, indices] = model.encode(x)
    return z,indices

def output_image_codes_impl(input_file,output_file,size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    img = PIL.Image.open(input_file)
    x_vqgan = preprocess(img.convert("RGB"), target_image_size=size)
    z,indices = convert_to_code(x_vqgan,model,device)
    with open(output_file,"w") as fp:
        for i in range(len(indices)):
            fp.write(str(indices[i].tolist()) + "\n")
    return model

def output_image_codes(params):
    input_file = params.input
    recons_output_file =  "codbook_recons_" + input_file.split("/")[-1]
    output_file =  "indices_" + input_file.split("/")[-1]
    output_file = ''.join(output_file.split(".")[:-1]) +  ".txt"
    model = output_image_codes_impl(input_file,output_file,params.size)
    gen_image_using_image_codebook_values(output_file,recons_output_file,model)
        


    
def gen_image_using_image_codebook_values(input_file,recons_output_file,model):
    arr = []
    with open(input_file) as fp:
        for line in fp:
            line = int(line.rstrip("\n"))
            arr.append(line)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = torch.LongTensor(arr)
    arr = arr.to(device)
    z = model.quantize.get_codebook_entry(arr,None)
    z = z.permute(1,0).contiguous().view(1,3,96,96)
    xrec = model.decode(z)
    xrec = custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    img.save(recons_output_file)

    

def main():
    parser = argparse.ArgumentParser(description='Codebook conversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="",help='Input file for some options')
    parser.add_argument('-output', action="store", dest="output",default="codebook",help='Output file/dir for codebook value images')
    parser.add_argument('-size', action="store", dest="size",default=384,type=int,help='Expected size. Do not change this default for VQ models')
    results = parser.parse_args()
    options = "Enter option:\n\t(1) generate image using custom codes. Input is text file of codes\n\t(2) Generate using custom codes. Input is numpy file\n\t(3) Output codebook indices for an input image\n\t"
    print(options)
    inp = int(input())
    if (inp == 1):
        gen_image_using_image_custom_codes(results)
    elif (inp == 2):
        gen_image_using_image_custom_codes2(results)
    else:
        output_image_codes(results)

if __name__ == '__main__':
    main()
        

