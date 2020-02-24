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
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from logger import Logger

logger = Logger().get_log()


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
    ret = os.listdir('data/')
    lst = []
    for item in ret:
        if item.endswith('csv') and 'corr' not in item and (not security_id or (security_id in item)):
            _df = pd.read_csv('data/{0}'.format(item))
            # index, exchangeCD, ticker, dataDate
            # barTime: 改成第几分钟
            bar_time_lst = _df['barTime']
            label5 = _df['label5']
            _df.drop(
                ['index', 'exchangeCD', 'ticker', 'dataDate', 'barTime', 'barTime.1', 'index.1', 'label5'],
                axis=1,
                inplace=True)
            _df = _df.apply(standadize, axis=0)
            bar_time_lst = _get_min(bar_time_lst)
            _df['barTime'] = bar_time_lst
            _df['label5'] = label5
            if not all_features:
                return _df
            if _df['label'][0] == 1.0 or ('barTime' not in _df.columns) or (_df['barTime'][0] != _df['barTime'][0]):
                logger.debug('verify data')
            lst.append(_df)
    df = pd.concat(lst)
    df.to_csv('data/all_features.csv')
    return df


def train_models(model_name='svc'):
    df = load_features(all_features=False)

    # targets = [2 if item > 0 else 1 if item == 0 else 0 for item in df['label5']] # 3 classes
    targets = [1 if item >= 0 else 0 for item in df['label5']]  # 2 classes
    # it seems the label here does not hv big influence to the results
    # df.drop(['label'], axis=1, inplace=True)
    data = df.values
    m = {
        'svc': svm.SVC(kernel='rbf', C=1),
        'adbc': AdaBoostClassifier(n_estimators=50)
    }.get(model_name)

    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=0)
    m.fit(x_train, y_train)
    # print(clf.score(x_test, y_test))
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    # scorer = make_scorer(f1_score, average='weighted')
    scorer = make_scorer(f1_score, average='micro')
    scores = cross_val_score(m, data, targets, cv=cv, scoring=scorer)
    print(scores)
    print(m.predict(data[:2]), targets[:2])


if __name__ == "__main__":
    train_models('svc')
