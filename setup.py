# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:10:57 2025

@author: fangwenji
"""

from setuptools import setup, find_packages

setup(
    name='WDPPN',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==2.3.0',
        'flash_attn==2.7.4.post1',
        'numpy==2.0.1',
        'scipy==1.15.1',
        'h5py==3.12.1',
        'tqdm==4.67.1',
        'einops==0.8.1',
        'omegaconf==2.3.0',
        'pytorch-lightning==1.8.6',
        'rich==13.9.4',
        'hydra-core==1.3.2',
        'opt_einsum==3.4.0',
        'detecta==0.0.5'
    ],
    python_requires='>=3.10, <3.11',
)
