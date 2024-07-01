# General Cost Neural Optimal Transport (GNOT)
Official Python implementation of **Neural Optimal Transport with General Cost Functionals** (ICLR 2024)(https://iclr.cc)

by Arip Asadulaev* Alexander Korotin*, Vage Egiazarian Petr Mokrov and Evgeny Burnaev.

## Structure
The implementation is GPU-based with the multi-GPU support. Tested with `torch== 1.9.0` and 1-4 Tesla V100.

The repository contains reproducible PyTorch source code for computing optimal transport maps and plans for general transport costs in high dimensions with neural networks. 
Examples are provided for toy problems (```gaussian_toy.ipynb, twomoons_toy.ipynb```) and for the class-guided (```dataset_transfer.ipynb, dataset_transfer_no_z.ipynb```), pair-guided image-to-image translation task (```paired_transport.ipynb```) and more. 

Pre-trained models are available in (https://drive.google.com/drive/folders/1ZK3t4fxJt5WjNYAuAa60NMTz1cp5IUgM?usp=share_link) 

## Citation
```
@inproceedings{asadulaev2022neural,
  title={Neural optimal transport with general cost functionals},
  author={Asadulaev, Arip and Korotin, Alexander and Egiazarian, Vage and Mokrov, Petr and Burnaev, Evgeny},
  booktitle={International Conference on Learning Representations},
  url={https://openreview.net/forum?id=gIiz7tBtYZ},
  year={2024}
}
```
