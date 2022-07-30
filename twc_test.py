import sys
import os
import requests
import math

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pdb
import codebook_conv
import argparse

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x_tensor = x
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x,x_tensor


import models_mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def compare_vectors(input_x,recons,mask):
    assert(input_x.shape == recons.shape)
    zero_dist_count = 0
    total_dist = 0
    count = 0
    sums = []
    for i in range(96):
        for j in range(96):
           if (mask[0,0,i,j] == 1):
               count += 1
               v1 = input_x[0,:,i,j] 
               v2 = recons[0,:,i,j] 
               euclidean = math.sqrt(((v1 -v2)*(v1-v2)).sum())
               sums.append(euclidean)
               if (euclidean == 0):
                   zero_dist_count += 1
               total_dist += euclidean
    total_count = 96*96
    mean = total_dist/total_count
    var = 0
    for i in range(len(sums)):
        var += (sums[i]  - mean)*(sums[i] - mean)
    var = math.sqrt(var/total_count)
    print("total masked count",count,"Average dist",round((total_dist/total_count),4), "STD", round(var,4), "total dist:",total_dist, " zero vec count ",zero_dist_count)
           

def test_vqmae(params):
    model = prepare_model(params.model, 'mae_vit_base_patch16')

    input_x = np.load(params.input)
    device = torch.device("cpu")
    vqmodel = codebook_conv.load_model(device)
    
    x_cpu = torch.from_numpy(input_x)
    # run MAE
    #the MAE model is loaded on cpu
    mask_value = params.mask
    print("mask value",mask_value)
    loss, y, mask = model(x_cpu.float(), mask_ratio=mask_value)
    recons = model.unpatchify(y)
    #y = vqmodel.quant_conv(y.to(device))
    #quant, emb_loss, info = vqmodel.quantize(y)
    #xrec = vqmodel.decode(quant)
    x_rec = vqmodel.decode(recons.to(device))
    xrec,x_tensor = custom_to_pil(x_rec[0])
    w, h = xrec.size[0],xrec.size[0]
    y = torch.tensor(x_tensor)
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    output_file = params.output + "_mask_" + str(mask_value) + ".png"
    img.save(output_file)

    

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    compare_vectors(input_x,recons.detach().numpy(),mask)
    
    changed = 0
    skipped = 0
    for i in range(96):
        for j in range(96):
            #print(i,j,":",mask[0,:,i,j])
            assert(mask[0,0,i,j] == mask[0,1,i,j] == mask[0,2,i,j])
            if (mask[0,0,i,j] == 1):
                changed += 1
                x_cpu[0,0,i,j] = recons[0,0,i,j]
                x_cpu[0,1,i,j] = recons[0,1,i,j]
                x_cpu[0,2,i,j] = recons[0,2,i,j]
            else:
                skipped += 1
                #print("skipping")
            
    print("Changed",changed,"Skipped",skipped,"Ratio",float(changed)/(float(changed) + float(skipped)))
    
    xrec = vqmodel.decode(x_cpu)
    xrec = codebook_conv.custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    x = xrec
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    output_file = params.output + "_visible_" + str(mask_value) + ".png"
    img.save(output_file)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vector quantized MAE test. Take an input image and reconstruct it after masking',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model",default='twc_models/train_checkpoint-1.pth',help='model path')
    parser.add_argument('-input', action="store", dest="input",required=True,help='Input file')
    parser.add_argument('-output', action="store", dest="output",default="converted",help='Output file')
    parser.add_argument('-mask', action="store", dest="mask",default=.05,type=float,help='Mask value')
    results = parser.parse_args()
    test_vqmae(results)
