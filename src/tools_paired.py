import pandas as pd
import numpy as np
#import lpips

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm_notebook as tqdm
from .fid_score import calculate_frechet_distance
from .distributions import PairedLoaderSampler, LoaderSampler
import torchvision.datasets as datasets
import h5py
from torch.utils.data import TensorDataset, ConcatDataset

import gc

from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder

def ema_update(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.data.copy_(beta*p_tgt.data + (1. - beta)*p_src.data)

def split_glued_image(im):
    w, h = im.size
    im_l, im_r = im.crop((0,0,w//2,h)), im.crop((w//2,0,w,h))
    return im_l, im_r

def paired_random_hflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
        im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)
    return im1, im2

def paired_random_vflip(im1, im2):
    if np.random.rand() < 0.5:
        im1 = im1.transpose(Image.FLIP_TOP_BOTTOM)
        im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)
    return im1, im2

def paired_random_rotate(im1, im2):
    angle = np.random.rand() * 360
    im1 = im1.rotate(angle, fillcolor=(255,255,255))
    im2 = im2.rotate(angle, fillcolor=(255,255,255))
    return im1, im2

def paired_random_crop(im1, im2, size):
    assert im1.size == im2.size, 'Images must have exactly the same size'
    assert size[0] <= im1.size[0]
    assert size[1] <= im1.size[1]
    
    x1 = np.random.randint(im1.size[0]-size[0])
    y1 = np.random.randint(im1.size[1]-size[1])
    
    im1 = im1.crop((x1, y1, x1 + size[0], y1 + size[1]))
    im2 = im2.crop((x1, y1, x1 + size[0], y1 + size[1]))
    
    return im1, im2

class PairedDataset(Dataset):
    def __init__(self, data_folder, labels_folder, transform=None, reverse=False, hflip=False, vflip=False, crop=None):
        self.transform = transform
        self.data_paths = sorted([
            os.path.join(data_folder,file) for file in os.listdir(data_folder)
            if (os.path.isfile(os.path.join(data_folder, file)) and file[-4:] in ['.png', '.jpg'])
        ])
        self.labels_paths = sorted([
            os.path.join(labels_folder,file) for file in os.listdir(labels_folder)
            if (os.path.isfile(os.path.join(labels_folder, file)) and file[-4:] in ['.png', '.jpg'])
        ])
        assert len(self.data_paths) == len(self.labels_paths)
        self.reverse = reverse
        self.hflip = hflip
        self.vflip = vflip
        self.crop = crop
        
    def __getitem__(self, index):
        x = Image.open(self.data_paths[index])
        y = Image.open(self.labels_paths[index])
        if self.crop is not None:
            x, y = paired_random_crop(x, y, size=self.crop)
        if self.hflip:
            x, y = paired_random_hflip(x,y)
        if self.vflip:
            x, y = paired_random_vflip(x,y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return (x,y,) if not self.reverse else (y,x,)

    def __len__(self):
        return len(self.data_paths)

class GluedDataset(Dataset):
    def __init__(self, path, transform=None, reverse=False, hflip=False, vflip=False, crop=None, rotate=False):
        self.path = path
        self.transform = transform
        self.data_paths = [
            os.path.join(path,file) for file in os.listdir(path)
            if (os.path.isfile(os.path.join(path, file)) and file[-4:] in ['.png', '.jpg'])
        ]
        self.reverse = reverse
        self.hflip = hflip
        self.vflip = vflip
        self.crop = crop
        self.rotate = rotate
        
    def __getitem__(self, index):
        xy = Image.open(self.data_paths[index])
        x, y = split_glued_image(xy)
        if self.reverse:
            x, y = y, x
        if self.crop is not None:
            x, y = paired_random_crop(x, y, size=self.crop)
        if self.hflip:
            x, y = paired_random_hflip(x,y)
        if self.vflip:
            x, y = paired_random_vflip(x,y)
        if self.rotate:
            x, y = paired_random_rotate(x,y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x,y

    def __len__(self):
        return len(self.data_paths)

def load_paired_dataset(name, path, img_size=64, batch_size=64, device='cuda', reverse=False, load_ambient=False):
    transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if name in ['edges2shoes', 'edges2handbags', 'anime-sketch']:
        hflip = True if (name != 'edges2shoes') else False
        train_set = GluedDataset(os.path.join(path, 'train'), transform=transform, reverse=reverse, hflip=hflip)
        test_set = GluedDataset(os.path.join(path, 'val'), transform=transform, reverse=reverse, hflip=hflip)
    elif name in ['facades', 'maps', 'cityscapes']:
        crop = (300,300) if name == 'maps' else None
        vflip = False if name in ['facades', 'cityscapes'] else True
        train_set = GluedDataset(os.path.join(path, 'train'), transform=transform, reverse=reverse, hflip=True, vflip=vflip, crop=crop)
        test_set = GluedDataset(os.path.join(path, 'val'), transform=transform, reverse=reverse, hflip=True, vflip=vflip, crop=crop)
    elif name == 'gta5_legend_map':
        input, target = name.split('_')[1:]
        train_set = PairedDataset(
            os.path.join(path, input, 'train'), os.path.join(path, target, 'train'),
            transform=transform, reverse=reverse, hflip=True, vflip=True,
        )
        test_set = PairedDataset(
            os.path.join(path, input, 'test'), os.path.join(path, target, 'test'),
            transform=transform, reverse=reverse, hflip=True, vflip=True,
        )
    elif name in ['comic_faces', 'comic_faces_v1', 'celeba_mask', 'aligned_anime_faces_sketch', 'safebooru_sketch']:
        if name == 'comic_faces':
            source_folder, target_folder = 'faces', 'comics'
        elif name == 'comic_faces_v1':
            source_folder, target_folder = 'face', 'comics'
        elif name == 'celeba_mask':
            source_folder, target_folder = 'CelebAMask-HQ-mask-color', 'CelebA-HQ-img'
        elif name == 'safebooru_sketch':
            source_folder, target_folder = 'safebooru_sketch', 'safebooru_jpeg'
        else:
            source_folder, target_folder = 'sketch', 'image'
        dataset = PairedDataset(
            os.path.join(path, source_folder), os.path.join(path, target_folder),
            transform=transform, reverse=reverse, hflip=True
        )
        idx = list(range(len(dataset)))
        test_ratio=0.1
        test_size = int(len(idx) * test_ratio)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
        train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    else:
        raise Exception('Unknown dataset')
        
    train_sampler = PairedLoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = PairedLoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler

def load_dataset(name, path, img_size=64, batch_size=64, device='cuda', load_ambient=False):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, img_size)
    elif name in ['celeba_female', 'celeba_male', 'aligned_anime_faces', 'comics', 'faces']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['dtd']:
        transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = ImageFolder(path, transform=transform)
    elif name in ['cartoon_faces']:
        transform = Compose([CenterCrop(420), Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['fruit360']:
        transform = Compose([
            Pad(14, fill=(255,255,255)), Resize((img_size, img_size)),
            ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = ConcatDataset((
            ImageFolder(os.path.join(path, 'Training'), transform=transform),
            ImageFolder(os.path.join(path, 'Test'), transform=transform)
        ))
    elif name in ['summer', 'winter', 'vangogh', 'photo']:
        if load_ambient:
            transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = Compose([
                RandomCrop(128),
                RandomHorizontalFlip(0.5),
                Resize((img_size, img_size)), 
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset = ImageFolder(path, transform=transform)
    else:
        raise Exception('Unknown dataset')
        
    if name in ['celeba_female', 'celeba_male']:
        with open('../datasets/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()[2:]
        if name == 'celeba_female':
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
    elif name in ['comics', 'faces']:
        idx = list(range(len(dataset)))
        if name == 'faces':
            idx = np.array(idx)[np.array(dataset.targets) == 1]
        else:
            idx = np.array(idx)[np.array(dataset.targets) == 0]
    else:
        idx = list(range(len(dataset)))
    
    test_ratio=0.1
    test_size = int(len(idx) * test_ratio)
    if name in ['summer', 'vangogh']:
        train_idx = np.array(idx)[np.array(dataset.targets) == 2]
        test_idx = np.array(idx)[np.array(dataset.targets) == 0]
    elif name in ['winter', 'photo']:
        train_idx = np.array(idx)[np.array(dataset.targets) == 3]
        test_idx = np.array(idx)[np.array(dataset.targets) == 1]
    elif name == 'fruit360':
        train_idx = idx[:len(dataset.datasets[0])]
        test_idx = idx[len(dataset.datasets[0]):]
    elif name == 'dtd':
        np.random.seed(0x000000); np.random.shuffle(idx)
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    else:
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
#     print(len(train_idx), len(test_idx))

    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler
import random

class GuidedDataset(Dataset):
    def __init__(self, dataset_in, dataset_out, num_labeled='all'):
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        assert len(dataset_in.classes) == len(dataset_out.classes) 
        self.num_classes = len(dataset_in.classes)
        self.subsets = [torch.where(dataset_out.targets == c)[0].numpy().tolist() for c in range(self.num_classes)]
        if num_labeled != 'all':
            assert type(num_labeled) == int
            self.subsets = [np.random.choice(subset, num_labeled) for subset in self.subsets]
        
    def __getitem__(self, index):
        x, c1 = self.dataset_in[index]
        y, c2 = self.dataset_out[random.choice(self.subsets[c1])]
        assert c1 == c2
        return x,y

    def __len__(self):
        return len(self.dataset_in)
    
# dataset = GuidedDataset(mnist_trainset, fashionmnist_trainset)

def load_guided_dataset(name, path, img_size=64, num_labeled='all', batch_size=64, device='cuda', reverse=False):
    if name in ['mnist2fashion']:
        transform = Compose([
            Resize((img_size, img_size)), ToTensor(),
            Normalize((.5), (.5)), Lambda(lambda x: -x.repeat(3,1,1)),
        ])
        mnist_train = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        fashion_train = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        fashion_test = datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        
        if not reverse:
            train_set = GuidedDataset(mnist_train, fashion_train, num_labeled=num_labeled)
            test_set = GuidedDataset(mnist_test, fashion_test)
        else:
            train_set = GuidedDataset(fashion_train, mnist_train, num_labeled=num_labeled)
            test_set = GuidedDataset(fashion_test, mnist_test)
    else:
        raise Exception('Unknown dataset')
       
    train_sampler = PairedLoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = PairedLoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

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

def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')    

    return TensorDataset(dataset, torch.zeros(len(dataset)))

def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False, use_Y=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in tqdm_notebook(range(n_epochs)):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if not use_Y:
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    else:
                        batch = ((Y[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_pushed_loader_metrics(T, loader, batch_size=8, n_epochs=1, verbose=False, device='cuda'):
    loss_alex = lpips.LPIPS(net='alex').to(device)
    loss_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_mse = nn.MSELoss(reduction='none')
    loss_l1 = nn.L1Loss(reduction='none')
    freeze(loss_alex); freeze(loss_vgg);  freeze(T)
    
    metrics = dict(mse=loss_mse, l1=loss_l1, alex=loss_alex, vgg=loss_vgg)
    results = dict({metric: 0. for metric in metrics.keys()})
    
    size = 0
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                X, Y = X.to(device), Y.to(device)
                size += X.size(0)
                for metric, loss in metrics.items():
                    results[metric] += loss(T(X), Y).sum()
                    
    for metric, loss in metrics.items():
        results[metric] /= size

    gc.collect(); torch.cuda.empty_cache()
    return results

def get_pushed_loader_stats(T, loader, batch_size=8, n_epochs=1, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(X[start:end].type(torch.FloatTensor).to(device)).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_Z_pushed_loader_stats(T, loader, ZC=1, Z_STD=0.1, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False, resnet=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                if not resnet:
                    Z = torch.randn(len(X), ZC, X.size(2), X.size(3)) * Z_STD
                    XZ = torch.cat([X,Z], dim=1)
                else:
                    Z = torch.randn(len(X), ZC, 1, 1) * Z_STD
                    XZ = (X, Z)
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if not resnet:
                        batch = T(XZ[start:end].type(torch.FloatTensor).to(device)).add(1).mul(.5)
                    else:
                        batch = T(
                            XZ[0][start:end].type(torch.FloatTensor).to(device),
                            XZ[1][start:end].type(torch.FloatTensor).to(device)
                        ).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def energy_distance(X1, X2, X3, Y1, Y2, Y3):
    assert X1.shape == X2.shape
    assert X3.shape == Y3.shape
    assert Y1.shape == Y2.shape
    assert len(X1.shape) == 2
    assert len(X3.shape) == 2
    assert len(Y1.shape) == 2
    ED = np.linalg.norm(X3-Y3, axis=1).mean() - \
    .5*np.linalg.norm(X1-X2, axis=1).mean() - \
    .5*np.linalg.norm(Y1-Y2, axis=1).mean()
    return ED

def EnergyDistances(T, XY_sampler, size=1048, batch_size=8, device='cuda'):
    assert size % batch_size == 0
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    freeze(model); freeze(T)
    
    pred_arr = []
    pixels = []
    
    with torch.no_grad():
        num_batches = size // batch_size
        for j in range(6 * num_batches):
            X, Y = XY_sampler.sample(batch_size)
            batch = T(X) if j < 3 * num_batches else Y
            img_size = batch.shape[2] * batch.shape[3]
            
            # inception stats
            pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(batch_size, -1))
            
            # color stats
            pixel_idx = np.random.randint(0, img_size, size)
            for k in range(batch_size):
                pixels.append(batch[k].flatten(start_dim=1)[:,pixel_idx[k]].cpu().numpy())
    
    # EID
    pred_arr = np.vstack(pred_arr)
    X1, X2, X3 = pred_arr[:size], pred_arr[size:2*size], pred_arr[2*size:3*size]
    Y1, Y2, Y3 = pred_arr[-3*size:-2*size], pred_arr[-2*size:-size], pred_arr[-size:]
    EID = energy_distance(X1, X2, X3, Y1, Y2, Y3)
    
    # ECD
    pixels = np.array(pixels)
    X1, X2, X3 = pixels[:size], pixels[size:2*size], pixels[2*size:3*size]
    Y1, Y2, Y3 = pixels[-3*size:-2*size], pixels[-2*size:-size], pixels[-size:]
    ECD = energy_distance(X1, X2, X3, Y1, Y2, Y3)
    
    gc.collect(); torch.cuda.empty_cache()
    return EID, ECD

# def EnergyColorDistance(T, XY_sampler, size=2048, batch_size=8, device='cuda'):
#     assert size % batch_size == 0
    
#     pred_arr = []
#     with torch.no_grad():
#         num_batches = size // batch_size
#         for j in range(6 * num_batches):
#             X, Y = XY_sampler.sample(batch_size)
#             batch = T(X) if j < 3 * num_batches else Y
#             img_size = batch.shape[2] * batch.shape[3]
#             batch = batch.reshape(batch_size, 3, -1)
#             pixel_idx = np.random.randint(0, img_size, size)
#             for k in range(batch_size):
#                 pred_arr.append(batch[k,:,pixel_idx[k]].cpu().numpy())
                
#     pred_arr = np.array(pred_arr)
#     X1, X2, X3 = pred_arr[:size], pred_arr[size:2*size], pred_arr[2*size:3*size]
#     Y1, Y2, Y3 = pred_arr[-3*size:-2*size], pred_arr[-2*size:-size], pred_arr[-size:]
#     ECD = energy_distance(X1, X2, X3, Y1, Y2, Y3)
            
#     gc.collect(); torch.cuda.empty_cache()
#     return ECD