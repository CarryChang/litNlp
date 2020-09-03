# -*- coding: utf-8 -*-
import setuptools
import io
import os
import platform
import subprocess
import sys
with io.open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
REQUIRED_PACKAGES = [
    'h5py', 'requests'
]
setuptools.setup(
  name="litNlp",
  version="0.8.5",
  packages=['litNlp', 'litNlp.model_structure'],
  author="CarryChang",
  author_email="coolcahng@gmail.com",
  url='https://github.com/CarryChang/litNlp',
  license='https://www.apache.org/licenses/LICENSE-2.0',
  include_package_data=True,
  description='A fast tool for sentiment analysis model with tensorflow2.0 ',
  # long_description='litNlp 是基于 Tensorflow2.0 实现的一个轻量级的深度情感极性推理模型，可以实现细粒度的多级别情感极性训练和预测。 GPU 和 CPU 平台通用，是搭建 NLP 分类模型类 baseline 的快速方案。'
  long_description=long_description,
  long_description_content_type='text/markdown',
  install_requires=REQUIRED_PACKAGES,
  python_requires=">=3.5",
  zip_safe=True,
  classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
  extras_require={
        "cpu": ["tensorflow>=2.0.1"],
        "gpu": ["tensorflow-gpu>=2.0.1"],
    },
  entry_points={
    },
  keywords=['text classification', 'nlp','batch predict',
              'deep learning', 'tensorflow', 'ml',],
)