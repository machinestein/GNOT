import pandas as pd
import numpy as np

import os
import itertools
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from .fid_score import calculate_frechet_distance
from .distributions import PairedLoaderSampler, LoaderSampler
import torchvision
import torchvision.datasets as datasets
import h5py
from torch.utils.data import TensorDataset
from .mnistm_utils import MNISTM
from PIL import Image
import random

import gc

from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Lambda
from torchvision.datasets import ImageFolder

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def get_stats(Z_sampler, size, batch_size=8, inception=False, resize_to_64=False, verbose=False):
    if inception:
        return get_inception_stats(Z_sampler, size, batch_size=batch_size, resize_to_64=resize_to_64, verbose=verbose)
    
    dims = np.prod(Z_sampler.sample(1)[0].shape)
    pred_arr = np.empty((size, dims))
   
    with torch.no_grad():
        for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
            start, end = i, min(i + batch_size, size)

            batch = ((Z_sampler.sample(end-start) + 1) / 2).type(torch.FloatTensor).cuda()
            pred = batch
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

class SumSequential(nn.Module):
    def __init__(self, G, Ts, alphas):
        super(SumSequential, self).__init__()
        self.G = G
        self.Ts = nn.ModuleList(Ts)
        self.alphas = alphas
        
    def forward(self, input):
        G_input = self.G(input)
        out = torch.zeros_like(G_input)
        for alpha, T in zip(self.alphas, self.Ts):
            out += alpha * T(G_input)
        return out

def score_gen(benchmark, G, Z_sampler, score_size=100000):
    assert benchmark.gauss_bar_sampler != None
    
    Z = Z_sampler.sample(score_size)
    with torch.no_grad():
        G_Z = G(Z).cpu().detach().numpy()
    G_Z_cov = np.cov(G_Z.T)
    G_Z_mean = np.mean(G_Z, axis=0)   
    BW2_UVP_G = 100 * calculate_frechet_distance(
        G_Z_mean, G_Z_cov,
        benchmark.gauss_bar_sampler.mean, benchmark.gauss_bar_sampler.cov,
    ) / benchmark.gauss_bar_sampler.var
        
    return BW2_UVP_G

def get_generated_stats_extended(T, Z_sampler, size, batch_size=8, inception=False, verbose=False, vae=False, ZD=128, Z_STD=1.0, name=None):              
    if 'Weak' in name:
        with torch.no_grad():
            X = Z_sampler.sample(1)
            Z = torch.randn(X.size(0), 1, ZD, 1, 1, device='cuda') * 1.
            XZ = (X, Z.flatten(start_dim=0, end_dim=1))
            out = T(*XZ) 
    else:
        with torch.no_grad():
            X = Z_sampler.sample(1)
            out = T(X)
    dims = np.prod(out[0].shape) 
    freeze(T); pred_arr = np.empty((size, dims))
    with torch.no_grad():
        for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
            start, end = i, min(i + batch_size, size)
            if 'Weak' in name:
                with torch.no_grad():
                    X = Z_sampler.sample(end-start)
                    Z = torch.randn(end-start, 1, ZD, 1, 1, device='cuda') * Z_STD
                    XZ = (X, Z.flatten(start_dim=0, end_dim=1))
                    out = T(*XZ) 
                    
            else:
                with torch.no_grad():
                    X = Z_sampler.sample(end-start)
                    out = T(X)
                    
            batch = ((out + 1) / 2).type(torch.FloatTensor).cuda()
            if inception:
                pred = model(batch)[0]
            else:
                pred = batch
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def test_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            outputs = model(x.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y.cuda()).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network:', accuracy)
    return accuracy

def compute_transport_accuracy(T, labeled_X_sampler, classifier, ZD=128, Z_STD =1.0, name=False):
    transport_results = []
    real_labels = []
    flat_transport_results =[]
    flat_real_labels = []
    for X, labels in labeled_X_sampler:
        real_labels.append(labels)
        if 'Weak' in name:
            with torch.no_grad():
                Z = torch.randn(10, 1, ZD, 1, 1, device='cuda') * Z_STD
                XZ = (X.cuda(), Z.flatten(start_dim=0, end_dim=1))
                T_X = T(*XZ)
                transport_results.append(T_X)
        else:
            with torch.no_grad():
                T_X = T(X.cuda())
                transport_results.append(T_X)

    for sublist, ys in zip(transport_results, real_labels):
        for item, y in zip(sublist, ys):
            flat_transport_results.append(item.data.cpu().numpy())
            flat_real_labels.append(y)

    flat_transport_results = torch.Tensor(flat_transport_results)
    flat_real_labels = torch.LongTensor(flat_real_labels)
    transport_dataset = torch.utils.data.TensorDataset(flat_transport_results, flat_real_labels)
    transport_loader = torch.utils.data.DataLoader(transport_dataset, batch_size=100, shuffle=False,
            num_workers=40, pin_memory=True)
    accuracy = test_accuracy(classifier, transport_loader)
    return accuracy