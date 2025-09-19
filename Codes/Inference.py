#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusion import GaussianDiffusion
from dataloader import CSIDataset
from denoise import DiffusionModel
from torch.utils.data import DataLoader
import numpy as np
import os


batchsize = 64
device = "cuda:1"



def InferenceFunction(trainfile,pretraindict,repeat_nums=1):

    if not os.path.exists("../GeneratedData"):
        os.mkdir("../GeneratedData")
    
    noise_schedule= np.linspace(1e-4, 1e-2, 100).tolist()

    trainset = CSIDataset(trainfile)
    trainloader = DataLoader(trainset,batchsize,shuffle=False,num_workers=4,pin_memory=True)

    denoiser = DiffusionModel().to(device)
    model = GaussianDiffusion(max_step=100,noise_schedule=noise_schedule,device=device).to(device)

    if pretraindict is not None:
        denoiser_dict = torch.load(pretraindict,map_location=device)
        denoiser.load_state_dict(denoiser_dict)
    else:
        print(" Not Find Pretrained Model !")
        exit()

    for repeat_time in range(repeat_nums):
        if not os.path.exists(os.path.join("../GeneratedData/","Generation_"+str(repeat_time)+"/")):
            os.mkdir(os.path.join("../GeneratedData/","Generation_"+str(repeat_time)+"/"))
        else:
            None
        total_count = 0
        for trainidx,(train_csi_amplitude,train_cond) in enumerate(trainloader):
            print(trainidx)
            train_csi_amplitude = train_csi_amplitude.to(device).float()
            train_cond = train_cond.to(device).float()
            B = train_csi_amplitude.shape[0]
            t = torch.randint(0, 100, [B], dtype=torch.int64).to(device)
            noised_t = model.degrade_fn(train_csi_amplitude,t)
            predict_t = denoiser(noised_t,t,train_cond) 
            predict_t = predict_t.squeeze(2) 
            predict_t = predict_t.transpose(2,1).contiguous()
            ba = predict_t.shape[0]
            for ii in range(ba):
                torch.save(predict_t[ii].clone().detach().to('cpu'),
                            os.path.join("../GeneratedData/","Generation_"+str(repeat_time)+"/",str(total_count)))
                total_count += 1

if __name__ == "__main__":
    InferenceFunction(trainfile="../datafiles/Train_7.txt",
                      pretraindict=os.path.join("../Models","epoch_best.pt"))






