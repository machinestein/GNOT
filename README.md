# General Cost Neural Optimal Transport (GNOT)
Official Python implementation of **Neural Optimal Transport with General Cost Functionals** (ICLR 2024)(https://iclr.cc)

by [Arip Asadulaev*](https://scholar.google.com/citations?user=wcdrgdYAAAAJ&hl=en), [Alexander Korotin*](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Vage Egiazarian](https://scholar.google.com/citations?user=Bktg6JEAAAAJ&hl=en), [Petr Mokrov](https://scholar.google.com/citations?hl=en&user=CRsi4IkAAAAJ) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

## Structure
The implementation is GPU-based with the multi-GPU support. Tested with `torch== 1.9.0` and 1-4 Tesla V100.

The repository contains reproducible PyTorch source code for computing optimal transport maps and plans for general transport costs in high dimensions using neural networks. 
Examples are provided in self-explanatory Jupyter notebooks (`notebooks/`):

- Toy problems with class-guided cost functional (```gaussian_toy.ipynb, twomoons_toy.ipynb```);
- Biology data Batch-effect problem solving with class-guided cost functional (```batch_effect.ipynb```);
- Image-to-image translation with class-guided cost functional (```dataset_transfer.ipynb, dataset_transfer_no_z.ipynb```);
- Image-to-image translation with pair-guided cost functional (```paired_transport.ipynb```). 

Pre-trained models are available in [Google Drive](https://drive.google.com/drive/folders/1ZK3t4fxJt5WjNYAuAa60NMTz1cp5IUgM?usp=share_link).

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
