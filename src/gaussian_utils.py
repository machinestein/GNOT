import os
import numpy as np
import random
import torch
from torch.utils.data import Subset, DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set_style("whitegrid")


def generate_data(num_mode=5, coordinates = [0,10], num_data_per_class=10000, sigma=0.15):
    total_data = {}
    fontsize = 20
    colums = np.linspace(coordinates[0], coordinates[1], num_mode)
    raws = np.linspace(coordinates[0], coordinates[1], num_mode)
    centers = []
    for col in colums:
        for raw in raws:
            centers.append([col,raw])
    colors = cm.rainbow(np.linspace(0, 1, len(centers)))
    for idx, mode, in enumerate(centers):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T
    
    return total_data

        
def plot_gaussian(model=None, num_mode=5, coordinates = [0,10], num_data_per_class=10000, sigma=0.15, labels = None, name = 'results', save_path=None):
    colums = np.linspace(coordinates[0], coordinates[1], num_mode)
    raws = np.linspace(coordinates[0], coordinates[1], num_mode)
    plt.figure(figsize=(10, 10), dpi=100)
    fontsize = 25
    centers = []
    for col in colums:
        for raw in raws:
            centers.append([col,raw])
    colors = cm.rainbow(np.linspace(0, 1, len(centers)))
    for idx, (mode, c) in enumerate(zip(centers, colors)):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        point = np.array([x, y]).T
        if model is not None:
            output = model(torch.from_numpy(point).float().cuda())
            output = output.detach().cpu().numpy().T
            sns.scatterplot(output[0], output[1], color=c)
        else:
            sns.scatterplot(x, y, color=colors[labels[idx]])
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3) 
    if save_path is not None:
        plt.savefig(save_path+'/{}'.format(name+'.png'))
    plt.show()
    
    
def build_dataloader(datadict):
    dataset = []
    labels = []
    for key, values in datadict.items():
        for value in values:
            dataset.append(value)
            labels.append(key)
    X, Y = torch.from_numpy(np.array(dataset)).float(), torch.from_numpy(np.array(labels)).long()
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    return dataset, loader