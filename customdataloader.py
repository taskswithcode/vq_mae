import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
from PIL import Image
import argparse

image_file_suffixes = ["jpg","jpeg","png"]

class VQDataLoader(Dataset):
    """Custom dataset loaded for transforming images to VQ codebook form"""

    def __init__(self, root_dir,output_base, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.output_base = output_base
        self.walk_dir_tree(root_dir)

    def walk_dir_tree(self,root_dir):
        self.files_list = []
        for (root, dirs, files) in os.walk(root_dir, topdown=True):
            print("The root is: ",root)
            #print(root)
            #print("The directories are: ")
            #print(dirs)
            #print("The files are: ")
            #print(files)
            #print('--------------------------------')
            for file_name in files:
                suffix = file_name.split('.')[-1].lower()
                if (suffix in image_file_suffixes):
                    self.files_list.append(root + "/" + file_name)
        print(len(self.files_list))


    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
        #                                self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        #if self.transform:
        #    sample = self.transform(sample)

        output = self.output_base + "/" + self.files_list[idx]
        dirs = '/'.join(output.split("/")[:-1])
        if (os.path.exists(dirs) == False):
            os.makedirs(dirs)
        sample = {'input': self.files_list[idx], 'index': str(idx),"output_dir":dirs}
        return sample

def test_loading(results):
    input_dir = results.input
    custom_dataset = VQDataLoader(root_dir=input_dir,output_base=results.output)

    for i in range(len(custom_dataset)):
        sample = custom_dataset[i]
        print(sample["input_dir"],sample["file"],sample["output_dir"])
        break

        
        
        #print(i, sample['image'].shape, sample['landmarks'].shape)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom dataset loader ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input", required=True,help='Directory of images')
    parser.add_argument('-output', action="store", dest="output", required=True,help='Output Directory of transformed images')
    results = parser.parse_args()

    test_loading(results)
