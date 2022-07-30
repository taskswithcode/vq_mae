import torch
import torchvision.transforms as T
from PIL import Image

def custom_trans(input_file,output_file,height,width):
    trans = T.Compose([
        T.RandomResizedCrop(size=(width, height),scale=(0.2,1.0),interpolation=3), # 3 is bicubic
        T.RandomHorizontalFlip()
      ])
    transformed_sample = trans(Image.open(input_file)).convert('RGB')
    transformed_sample.save(output_file)


custom_trans("converted_bird384.png","my1.png",384,384)
custom_trans("converted_bird384.png","my2.png",384,384)
custom_trans("converted_bird384.png","my3.png",384,384)
custom_trans("converted_bird384.png","my4.png",384,384)
custom_trans("converted_bird384.png","my5.png",384,384)

