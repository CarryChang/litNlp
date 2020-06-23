# -*- coding: utf-8 -*-
import setuptools
# with open("README.md", "r", encoding='utf-8') as fh:
#     long_description = fh.read()
REQUIRED_PACKAGES = [
    'h5py', 'requests'
]
setuptools.setup(
  name="litNlp",
  version="0.8.2",
  packages=['litNlp', 'litNlp.model_structure'],
  author="CarryChang",
  author_email="coolcahng@gmail.com",
  url='https://github.com/CarryChang/litNlp',
  include_package_data=True,
  description='A fast tool for text classification model with tensorflow ',
  long_description='litNlp是基于Tensorflow2.0实现的一个轻量级的深度文本分类模型,支持多分类，并默认二分类，是搭建文本分类模型的快速方案',
  install_requires=['tensorflow==2.0.1', 'numpy==1.17.0', 'pandas', 'sklearn'],
  python_requires=">=3.4",
  zip_safe=True,
  classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
  extras_require={
        "cpu": ["tensorflow==2.0.0"],
        "gpu": ["tensorflow-gpu==2.0.0"],
    },
  entry_points={
    },
  license="Apache-2.0",
  keywords=['text classification', 'nlp','batch predict',
              'deep learning', 'tensorflow', 'ml',],
)