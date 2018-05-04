#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/4 上午9:45
# @Author  : Shiloh Leung
# @Site    : 
# @File    : setup.py
# @Software: PyCharm
from setuptools import setup, find_packages

setup(
    name='OrTensor-Numba',
    version='0.1',
    author='Siqi Liang',
    author_email='zszxlsq@gmail.com',
    packages=find_packages(exclude=('tests')),
    install_requires=['numpy', 'numba']
)
