import os
import numpy as np
import random
import torch
from torch.utils.data import Subset, DataLoader, Dataset

class Sampler:
    def __init__(
        self, device='cuda'):
        self.device = device
    
    def sample(self, size=5):
        pass
    

class PairedSubsetSampler(Sampler):
    def __init__(self, loader, subsetsize = 8, weight=None, device='cuda'):
        super(PairedSubsetSampler, self).__init__(device)
        self.loader = loader
        self.subsetsize = subsetsize
        if weight is None:
            weight = [1/self.loader.num_classes for _ in range(self.loader.num_classes)]
        self.weight = weight
        
    def sample(self, batch_size=5):
        classes = np.random.choice(self.loader.num_classes, batch_size, p=self.weight)
        batch_X = []
        batch_Y = []
        with torch.no_grad():
            for class_ in classes: 
                X, Y = self.loader.get(class_, self.subsetsize)
                batch_X.append(X.clone().to(self.device).float())
                batch_Y.append(Y.clone().to(self.device).float())
                
        return torch.stack(batch_X).to(self.device), torch.stack(batch_Y).to(self.device)
    
    
class SubsetGuidedDataset(Dataset):
    def __init__(self, dataset_in, dataset_out, num_labeled='all', in_indicies=None, out_indicies=None):
        super(SubsetGuidedDataset, self).__init__()
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        assert len(in_indicies) == len(out_indicies) 
        self.num_classes = len(in_indicies)
        self.subsets_in = in_indicies
        self.subsets_out = out_indicies
        if num_labeled != 'all':
            assert type(num_labeled) == int
            self.subsets_out = [np.random.choice(subset, num_labeled) for subset in  self.subsets_out]
        
    def get(self, class_, subsetsize):
        x_subset = []
        y_subset = []
        in_indexis = random.sample(list(self.subsets_in[class_]),subsetsize)
        out_indexis = random.sample(list(self.subsets_out[class_]),subsetsize)
        for x_i, y_i in zip(in_indexis, out_indexis):
            x, c1 = self.dataset_in[x_i]
            y, c2 = self.dataset_out[y_i]
            assert c1 == c2
            x_subset.append(x)
            y_subset.append(y)
        return torch.stack(x_subset), torch.stack(y_subset)

    def __len__(self):
        return len(self.dataset_in)


def get_indicies_subset(dataset,  new_labels = {}, classes=4, subset_classes=None):
    labels_subset = []
    dataset_subset = [] 
    class_indicies = [[] for _ in range(classes)]
    i = 0
    for x,y in dataset:
        if y in subset_classes:
            if type(y)== int:
                class_indicies[new_labels[y]].append(i)
                labels_subset.append(new_labels[y])
            else:
                class_indicies[new_labels[y.item()]].append(i)
                labels_subset.append(new_labels[y.item()])
            dataset_subset.append(x)
            i+=1
    return dataset_subset, labels_subset, class_indicies