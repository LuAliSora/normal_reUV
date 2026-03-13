import torch
from torch.utils import data

from pathlib import Path

# import torch.nn.functional as F
# import torchvision.transforms as transforms

# from PIL import Image
            

class ImgSet(data.Dataset):

    def __init__(self, imgFolder:Path, resFolder:Path):
        self.inputDir=imgFolder

        # print(self.imgList)
        resFolder.mkdir(parents=True,exist_ok=True)
        self.outputDir=resFolder

        imgList=[img.name for img in imgFolder.glob('*.jpg')]
        resList=[img.name for img in f.glob('*.jpg')]
        self.genList=list(set(imgList) - set(resList))

    def __getitem__(self, index):
        img=self.genList[index]
        imgPath=(self.inputDir)/img
        resPath=(self.outputDir)/img
        return imgPath, resPath
    
    def __len__(self):
        return len(self.genList)
    


            
