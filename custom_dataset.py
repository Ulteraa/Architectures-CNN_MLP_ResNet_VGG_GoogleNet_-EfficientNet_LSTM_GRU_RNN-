import os.path
import pandas as pd
import torch
import csv
from torch.utils.data import  Dataset

import torchvision
from skimage import io
def read_(dir):
    with open(dir, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        list_=[]
        for row in spamreader:
            r=row[0].split(',')
            list_.append(r)
    return list_
class Custom_dataset(Dataset):
    def __init__(self,dir,cvs_file,transform):
        self.anotation=read_(cvs_file)
        self.dir=dir
        self.transform=transform
    def __len__(self):
        return len(self.anotation)
    def __getitem__(self, index):

        im_path=''.join([self.dir,self.anotation[index][0]])
        print(im_path)

        target=torch.tensor(int(self.anotation[index][1]))
        image=io.imread(im_path)
        if self.transform:
            image=self.transform(image)
        return (image,target)


