import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import freeze, compute_transport_accuracy

import torch
import gc

def plot_images(X, Y, T, Z=None):
    freeze(T);
    with torch.no_grad():
        if Z is not None:
            XZ = (X,Z)
            T_X = T(*XZ)
        else:
            T_X = T(X)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)
    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    fig.tight_layout(pad=0.001)
    return fig, axes

def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)

def plot_random_paired_images(XY_sampler, T):
    X, Y = XY_sampler.sample(10)
    return plot_images(X, Y, T)


def plot_Z_images(XZ, Y, T, resnet=False):
    freeze(T);
    with torch.no_grad():
        if not resnet:
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).permute(1,2,3,0).reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4).permute(4,3,0,1,2).flatten(start_dim=0, end_dim=1)
        else:
            T_XZ = T(
                *(XZ[0].flatten(start_dim=0, end_dim=1), XZ[1].flatten(start_dim=0, end_dim=1))
            ).permute(1,2,3,0).reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4).permute(4,3,0,1,2).flatten(start_dim=0, end_dim=1)
        if not resnet:
            imgs = torch.cat([XZ[:,0,:Y.shape[1]], T_XZ, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)
        else:
            imgs = torch.cat([XZ[0][:,0], T_XZ, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(4):
        axes[i+1, 0].set_ylabel('T(X,Z)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_Z_images(X_sampler, ZC, Z_STD, Y_sampler, T, resnet=False):
    X = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        if not resnet:
            Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device='cuda') * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        else:
            Z = torch.randn(10, 4, ZC, 1, 1, device='cuda') * Z_STD
            XZ = (X, Z,)
    Y = Y_sampler.sample(10)
    return plot_Z_images(XZ, Y, T, resnet=resnet)

def plot_random_paired_Z_images(XY_sampler, ZC, Z_STD, T, resnet=False):
    X, Y = XY_sampler.sample(10)
    X = X[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        if not resnet:
            Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device='cuda') * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        else:
            Z = torch.randn(10, 4, ZC, 1, 1, device='cuda') * Z_STD
            XZ = (X, Z,)
    return plot_Z_images(XZ, Y, T, resnet=resnet)
