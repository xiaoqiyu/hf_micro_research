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
    # if file_name:
    #     return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'data', file_name))
    # return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'data'))
    _base_dir = os.path.join(os.path.realpath(os.path.pardir))
    if file_name:
        # return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models', model_name))
        return os.path.realpath(os.path.join(_base_dir.split('hf_research')[0] + 'hf_research', 'data', file_name))
    # return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models'))
    return os.path.realpath(os.path.join(_base_dir.split('hf_research')[0] + 'hf_research', 'data'))


def get_full_model_path(model_name=None):
    _base_dir = os.path.join(os.path.realpath(os.path.pardir))
    if model_name:
        # return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models', model_name))
        return os.path.realpath(os.path.join(_base_dir.split('hf_research')[0] + 'hf_research', 'models', model_name))
    # return os.path.realpath(os.path.join(os.path.realpath(os.path.pardir), 'models'))
    return os.path.realpath(os.path.join(_base_dir.split('hf_research')[0] + 'hf_research', 'models'))
