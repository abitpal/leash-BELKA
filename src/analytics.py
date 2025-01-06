import matplotlib.pyplot as plt
import wandb
import torch
import numpy as np

def collect(_type, var):
    if _type == "train" and var['i']%10 == 0:
        loss = var['loss']
        output: torch.Tensor = var['out'].cpu().detach()
        true: torch.Tensor = var['bind'].cpu().detach()
        true_nump = true.numpy()
        out_nump = output.numpy()
        print(np.concatenate((true_nump, out_nump), axis=-1))
        print("Mean")
        print(out_nump.mean())
        print(true_nump.mean())
        if (var['i'] == 0):
            print("EPOCH: " + str(var['epoch']))
    pass