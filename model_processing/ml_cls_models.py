#!/user/bin/env python
# coding=utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: ml_cls_models.py
@time: 19-11-15 下午4:03
@desc:
'''

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from utils.helper import get_full_data_path
from utils.helper import get_full_model_path
from utils.logger import Logger
from sklearn.externals import joblib

# todo will be added later
# from data_processing.hf_features import load_features

logger = Logger().get_log()


#
def standadize(arr):
    arr = arr.replace(np.inf, 0.0)
    arr = arr.replace(-np.inf, 0.0)
    _max, _min = arr.max(), arr.min()
    return (arr - _min) / (_max - _min)


def _get_min(bartime_lst):
    ret = []
    for item in bartime_lst:
        _h, _m = item.split(':')
        ret.append((int(_h) - 9) * 60 + int(_m))
    _max, _min = max(ret), min(ret)
    return [(item - _min) / (_max - _min) for item in ret]


def load_features(all_features=False, security_id=None):
    ret = os.listdir(get_full_data_path())
    lst = []
    for item in ret:
        if item.endswith('csv') and 'corr' not in item and (not security_id or (item.startswith(security_id))):
            _df = pd.read_csv(get_full_data_path(item))
            # index, exchangeCD, ticker, dataDate
            # barTime: change to the offset minutes since the market start, and the relative time span in the day
            bar_time_lst = _df['barTime']
            # label5 = _df['label5']
            label = _df['label']
            _del_col = list(set(_df.columns).intersection(
                {'index', 'exchangeCD', 'ticker', 'dataDate', 'barTime', 'barTime.1', 'index.1', 'label5', 'label',
                 'ma20', 'maDiff20'}))
            _df.drop(
                _del_col,
                axis=1,
                inplace=True)
            _df = _df.apply(standadize, axis=0)
            bar_time_lst = _get_min(bar_time_lst)
            _df['barTime'] = bar_time_lst
            # _df['label5'] = label5
            _df['label'] = label
            if not all_features:
                return _df
            if _df['label'][0] == 1.0 or ('barTime' not in _df.columns) or (_df['barTime'][0] != _df['barTime'][0]):
                logger.debug('verify data')
            lst.append(_df)
    df = pd.concat(lst)
    all_feature_path = get_full_data_path('all_features_{0}.csv'.format(security_id))
    df.to_csv(all_feature_path)
    return df


def train_models(*args, **kwargs):
    model_name = args[0]
    security_id = kwargs.get('security_id')
    n_split = kwargs.get('n_split') or 5
    test_ratio = kwargs.get('test_ratio') or 0.3
    kernal = kwargs.get('kernel') or 'rbf'

    df = load_features(all_features=False, security_id=security_id)

    # targets = [2 if item > 0 else 1 if item == 0 else 0 for item in df['label5']] # 3 classes
    targets = [1 if item >= 0 else 0 for item in df['label']]  # 2 classes
    # it seems the label here does not hv big influence to the results
    df.drop(['label'], axis=1, inplace=True)
    logger.info('feature shape is:{0}'.format(df.shape))
    data = df.values
    m = {
        'svc': svm.SVC(kernel=kernal, C=1),
        'adbc': AdaBoostClassifier(n_estimators=50)
    }.get(model_name)

    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=0)
    m.fit(x_train, y_train)
    # print(clf.score(x_test, y_test))
    cv = ShuffleSplit(n_splits=n_split, test_size=test_ratio, random_state=0)
    # scorer = make_scorer(f1_score, average='weighted')
    scorer = make_scorer(f1_score, average='micro')
    scores = cross_val_score(m, x_train, y_train, cv=cv, scoring=scorer)
    model_path = get_full_model_path('{0}'.format(model_name))
    joblib.dump(m, model_path, protocol=2)
    m1 = joblib.load(model_path)
    test_score = m1.score(x_test, y_test)
    logger.info(
        'Train results for {0},{1} is: cv:{2}, out of sample score:{3}'.format(args, kwargs, scores, test_score))


if __name__ == "__main__":
    train_models('svc', security_id='002415.XSHE', n_split=5, test_rtaio=0.3, kernel='rbf')
