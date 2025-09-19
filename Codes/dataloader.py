#encoding=utf-8

import torch
import torch.utils
from torch.utils.data import Dataset,DataLoader
import numpy as np


class CSIDataset(Dataset):
    def __init__(self,fileName,measurement_size=1):
        self.measurement_size = measurement_size
        self.filenamelist = []
        self.sizelist = []
        self.centerxlist = []
        self.centerylist = []
        with open(fileName,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenamelist.append(line[0])
                self.sizelist.append(line[1])
                self.centerxlist.append(line[2])
                self.centerylist.append(line[3])
    
    def __getitem__(self,index):
        segment = self.filenamelist[index*self.measurement_size:(index+1)*self.measurement_size]
        segmentsize = self.sizelist[index*self.measurement_size:(index+1)*self.measurement_size]
        segmentcenterx = self.centerxlist[index*self.measurement_size:(index+1)*self.measurement_size]
        segmentcentery = self.centerylist[index*self.measurement_size:(index+1)*self.measurement_size]

        csi_amplitude_list,cond_list = [],[]
        for ii in range(self.measurement_size):
            csi_data = torch.load(segment[ii])
            csi_data = csi_data.reshape(14,240,52)
            csi_data = csi_data.transpose(1,0)
            csi_data_amplitude = torch.abs(csi_data).float()
            size = float(segmentsize[ii])
            centerx = float(segmentcenterx[ii])
            centery = float(segmentcentery[ii])
            cond = torch.tensor([size,centerx,centery]).float()
            csi_amplitude_list.append(csi_data_amplitude)
            cond_list.append(cond)

        csi_amplitude = torch.stack(csi_amplitude_list,dim=1)
        cond = torch.stack(cond_list,dim=0)
        return csi_amplitude,cond
    
    def __len__(self):
        return len(self.filenamelist) // self.measurement_size
    

            
