# FeDLaS : Feature-Modulated Bidirectional Label Smoothing for Neural Network Calibration (ECCV'26)
Official PyTorch implementation of ["**FeDLaS : Feature-Modulated Bidirectional Label
Smoothing for Neural Network Calibration**"]([ARXIV LINK]), Thiru Thillai Nadarasar Bahavan, Sachith Seneviratne, and Saman Halgamuge.


> **Abstract:** *Deep Neural Network (DNN) classifiers suffer from poor calibration when their softmax outputs (predictive confidence) deviate from the empirical likelihoods. This manifests itself as either overconfident incorrect predictions or under-confident correct predictions. Label smoothing (LS) enhances model calibration by introducing entropy regularization during training through redistributing probability mass from the ground-truth label to the remaining classes. LS, including Margin-based LS (MbLS), have restrictive assumptions: they rely on predefined, uniform smoothing rules and only tackle overconfidence. In reality, samples exhibit diverse characteristics, such as difficulty/ambiguity, that interact with the evolving nature of the model being trained. In training, samples may have various degrees of under- or overconfidence. To overcome this, a mechanism that identifies the specific confidence state of each sample and determines the appropriate degree of smoothing in each training step is needed, tailoring the adjustment to the individual sample. We propose FeDLaS: Feature-Modulated Bidirectional Label Smoothing, a plug-and-play algorithm for label smoothing-based losses. In FeDLaS, we introduce a Feature Norm-based Confidence Indicator (NCI) to control smoothing and a Bidirectional Calibration Gating (BCG) module to detect both over and under-confidence. Our algorithm can be integrated with LS and MbLS based losses when applied to standard DNNs, leading to enchanced performance. Extensive experiments on standard and fine-grained high-resolution vision benchmarks show that FeDLaS consistently improves calibration compared to modern baselines, reducing Expected Calibration Error (ECE) and Adaptive ECE while maintaining Top-1 accuracy.*

<p align="center">
    <img src=./img.png width="800">
</p>

## 1. Requirements
### Environments
```bash
pip install -e .
```

### Datasets

All dataset root paths are centralised in one file — **edit this before running anything**:
```
configs/data/path_configs.yaml
```

#### CIFAR-10 / CIFAR-100
Downloaded automatically via torchvision on first run. Set `cifar_root` in `path_configs.yaml`.

#### Tiny-ImageNet
Download and extract, then set `tiny_imagenet_root` in `path_configs.yaml`. Expected layout:
```
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    │   ├── images/
    │   └── val_annotations.txt
    ├── wnids.txt
    └── words.txt
```

#### CUB-200-2011
Download from [the official site](https://www.vision.caltech.edu/datasets/cub_200_2011/) and set `cub_root` in `path_configs.yaml`.

#### FGVC-Aircraft
Download from [the official site](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and set `air_root` in `path_configs.yaml`.

#### OOD Datasets (SVHN)
Downloaded automatically via torchvision. Set `svhn_root` in `path_configs.yaml`.

## 2. Training & Evaluation

### CIFAR-10
```bash
# FEDLAS
python tools/train.py log_period=100 seed=1 loss=fedlas_cif10 model=resnet50_cifar10 --config-name train_cifar10.yaml

# FEDLAS+
python tools/train.py log_period=100 seed=1 loss=fedlasplus_cif10 model=resnet50_cifar10 --config-name train_cifar10.yaml
```

### CIFAR-100
```bash
# FEDLAS
python tools/train.py log_period=100 seed=1 loss=fedlas_cif100 model=resnet50_cifar100 --config-name train_cifar100.yaml

# FEDLAS+
python tools/train.py log_period=100 seed=1 loss=fedlasplus_cif100 model=resnet50_cifar100 --config-name train_cifar100.yaml
```

### Tiny-ImageNet
```bash
# FEDLAS
python tools/train.py log_period=100 seed=1 loss=fedlas_tiny model=resnet50_tiny --config-name train_tiny.yaml

# FEDLAS+
python tools/train.py log_period=100 seed=1 loss=fedlasplus_tiny model=resnet50_tiny --config-name train_tiny.yaml
```

### CUB
```bash
# FEDLAS
python tools/train.py log_period=100 seed=1 loss=fedlas_tiny --config-name train_vit_cub.yaml

# FEDLAS+
python tools/train.py log_period=100 seed=1 loss=fedlasplus_tiny  --config-name train_vit_cub.yaml
```

### AIR
```bash
# FEDLAS
python tools/train.py log_period=100 seed=1 loss=fedlas_tiny  --config-name train_vit_air.yaml

# FEDLAS+
python tools/train.py log_period=100 seed=1 loss=fedlasplus_tiny --config-name train_vit_air.yaml
```

### OOD Detection
The OOD dataset is selected via the `data/ood` config group (`svhn`, `cifar10`, or `cifar100`):
```bash
# CIFAR-10 (in) vs SVHN (out)
python tools/test.py task=ood data=cifar10 data/ood=svhn model=resnet50_cifar10 \
  hydra.run.dir=/path/to/output test.checkpoint=best.pth

# CIFAR-10 (in) vs CIFAR-100 (out)
python tools/test.py task=ood data=cifar10 data/ood=cifar100 model=resnet50_cifar10 \
  hydra.run.dir=/path/to/output test.checkpoint=best.pth

# CIFAR-100 (in) vs SVHN (out)
python tools/test.py task=ood data=cifar100 data/ood=svhn model=resnet50_cifar100 \
  hydra.run.dir=/path/to/output test.checkpoint=best.pth

# CIFAR-100 (in) vs CIFAR-10 (out)
python tools/test.py task=ood data=cifar100 data/ood=cifar10 model=resnet50_cifar100 \
  hydra.run.dir=/path/to/output test.checkpoint=best.pth
```

## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@inproceedings{
  title={FedLAS: Feature-Modulated Bidirectional Label Smoothing for Neural Network Calibration},
  author={Thiru Thillai Nadarasar Bahavan, Sachith Seneviratne, Saman Halgamuge},
  booktitle={19th European Conference on Computer Vision},
  year={2026}
}
```


## References
Our work is mainly built on [FLSD](https://github.com/torrvision/focal_calibration), [MbLS](https://github.com/by-liu/MbLS), and [ACLS](https://github.com/cvlab-yonsei/ACLS). Thanks to the authors!

