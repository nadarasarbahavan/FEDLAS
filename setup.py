from setuptools import setup, find_packages

setup(
    name="calibrate",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "albumentations==1.3.1",
        "opencv-python==4.10.0.84",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "terminaltables==3.1.10",
        "matplotlib==3.7.5",
        "wandb==0.17.0",
        "umap-learn==0.5.5",
        "timm==0.9.12",
        "tqdm==4.66.1",
        "Pillow==9.3.0",
    ],
)
