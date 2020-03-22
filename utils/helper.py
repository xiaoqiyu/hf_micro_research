#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: helper.py
@time: 2020/3/21 17:15
@desc:
'''
import os


def get_full_data_path(file_name=None):
    if file_name:
        return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'data', file_name))
    return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'data'))


def get_full_model_path(model_name=None):
    if model_name:
        return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models', model_name))
    return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models'))


