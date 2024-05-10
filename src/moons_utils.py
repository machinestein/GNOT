import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pylab as plt
import torch


def generate_moon_dataset(theta=60, n_samples=150, noise=0.1, n_pairs=5, use_pair=False, 
                          random_pair=33, random_state=45):

    def generate_moon(n_samples=100, noise=0.1, random_state=45, shuffle=False):
        x, y = make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)
        x[:, 0] = x[:, 0] - 0.5
        x[:, 1] = x[:, 1] - 0.25
        return x
    
    def rotate_moon(Xt, theta, x0=0, y0=0, r=1):
        theta = theta/180.*np.pi
        Xtt = np.zeros(Xt.shape)
        Xtt[:,0] = Xt[:,0] - x0
        Xtt[:,1] = Xt[:,1] - y0
        cos = Xtt[:,0]/r
        sin = Xtt[:,1]/r
        Xt[:,0] = x0 + Xtt[:,0]*np.cos(theta) - Xtt[:,1]*np.sin(theta)
        Xt[:,1] = y0 + Xtt[:,1]*np.cos(theta) + Xtt[:,0]*np.sin(theta)
        Xt = Xt + 0.1*np.random.rand(Xt.shape[0], Xt.shape[1])
        return Xt

    # Generate train and test data
    xs = generate_moon(n_samples=n_samples, noise=noise, random_state=random_state)
    xt = rotate_moon(xs.copy(), theta)
    
    np.random.seed(random_pair)
    num, dim = xs.shape
    if not use_pair:
        m = int(num/2)
        range_sub = [[range(m), range(m)], 
                          [range(m, num), range(m, num)]]
    else:
        inds_chosen = list(np.sort(np.random.choice(range(n_samples), n_pairs, replace=False)))
        inds_not_chosen = list(set(range(n_samples)) - set(inds_chosen))
        range_sub = [[[i], [i]] for i in inds_chosen] 
        if len(inds_not_chosen) > 0:
            range_sub += [[inds_not_chosen, inds_not_chosen]]
            
    n = xs.shape[0]
    ys = np.concatenate([np.zeros(int(n/2)), np.ones(int(n/2))]).astype(np.int)
    yt = ys.copy()
    return xs, ys, xt, yt, range_sub
    
    
def plot_twomoon(xs, xt, T=None, show_lines=False, name='results', save_path=None, use_Z=False):
    fontsize = 25
    colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    n = xs.shape[0]
    class_indicies = [list(range(0,int(n/2))), list(range(int(n/2),n))]
    source_colors = ['green','red']
    plt.figure(figsize=(10, 10), dpi=100)

    for color_index, class_ in enumerate(class_indicies):
        for i in class_:
            xs_i = xs[i]
            xt_shadow = xt[i]
            if T is not None:
                if use_Z:
                    X = torch.from_numpy(xs_i).float().cuda()
                    Z = torch.randn(2, device='cuda') * Z_STD
                    XZ = torch.cat([X, Z], dim=0)
                    output = T(XZ)
                    xt_i = output.detach().cpu().numpy()
                else:
                    X = torch.from_numpy(xs_i).float().cuda()
                    output = T(X)
                    xt_i = output.detach().cpu().numpy()
            else:
                xt_i = xt[i]
                
            if show_lines:
                plt.plot([xs_i[0], xt_i[0]], [xs_i[1], xt_i[1]], c='k', alpha=1, linewidth=0.2)
            plt.scatter(xs_i[0], xs_i[1], c=source_colors[color_index], marker='o', s=50,)
            plt.scatter(xt_i[0], xt_i[1], c=source_colors[color_index], marker='x', s=90,)
            
            plt.plot([xs_i[0], xt_shadow[0]], [xs_i[1], xt_shadow[1]], c='white', alpha=1, linewidth=0.00001)
    
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)    
    if save_path:
        plt.savefig(save_path+'/{}'.format(name+'.png'))
    plt.show() 