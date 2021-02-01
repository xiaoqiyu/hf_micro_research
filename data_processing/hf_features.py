#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: hf_features.py
@time: 19-11-15 下午4:03
@desc:
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import uqer
import pprint
from uqer import DataAPI
import seaborn as sns
from data_processing.gen_sample import get_samples
from utils.logger import Logger
from utils.helper import get_full_data_path
from utils.helper import get_full_data_path
from utils.helper import get_full_model_path
from account_info.get_account_info import get_account_info

sns.set()
logger = Logger().get_log()

uqer_client = uqer.Client(token=get_account_info().get('uqer_token'))

REMOVE_TICK_COLS = ['dataDate', 'exchangeCD', 'ticker', 'dataTime', 'barTime', 'shortNM', 'currencyCD', 'askPrice1',
                    'askPrice2',
                    'askPrice3', 'askPrice4', 'askPrice5', 'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4',
                    'askVolume5',
                    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5', 'bidVolume1', 'bidVolume2',
                    'bidVolume3',
                    'bidVolume4', 'bidVolume5']
REMOVE_MIN_COLS = ['unit', 'currencyCD', 'shortNM']


def _cal_tick_features(df):
    total_vol = list(df['volume'])[-1]
    # price
    df['avgPrice'] = df['value'] / df['volume']
    df['amplitude'] = (df['highPrice'] - df['lowPrice']) / df['lastPrice']
    df['spread'] = df['bidPrice1'] - df['askPrice1']
    df['openDiff'] = (df['openPrice'] - df['prevClosePrice']) / df['prevClosePrice']
    df['trackError'] = (df['lastPrice'] - df['avgPrice']) / df['avgPrice']
    df['askTrackError1'] = (df['askPrice1'] - df['avgPrice']) / df['avgPrice']
    df['bidTrackError1'] = (df['bidPrice1'] - df['avgPrice']) / df['avgPrice']
    df['midPrice'] = (df['askPrice1'] + df['bidPrice1']) / 2
    df['mktAskWidth'] = df['askPrice5'] - df['askPrice1']
    df['mktBidWidth'] = df['bidPrice5'] - df['bidPrice1']
    df['mktWidth'] = df['mktBidWidth'] - df['mktAskWidth']
    df['midSpread'] = df['midPrice'] - df['lastPrice']  # return
    df['ma5'] = df[['lastPrice']].rolling(5).mean()
    df['ma10'] = df[['lastPrice']].rolling(10).mean()
    df['ma20'] = df[['lastPrice']].rolling(20).mean()
    df['maDiff10'] = df['ma5'] - df['ma10']
    df['maDiff20'] = df['ma5'] - df['ma20']
    # TODO to add
    # df['smartPrice']  #A variation on mid-price where the average of the bid and ask prices is weighted according to their inverse volume
    # Trade Sign: A feature measuring whether buyers or sellers crossed the spread more frequently in recent executions

    # vol
    df['totalAskVolume'] = df['askVolume1'] + df['askVolume2'] + df['askVolume3'] + df['askVolume4'] + df['askVolume5']
    df['totalBidVolume'] = df['bidVolume1'] + df['bidVolume2'] + df['bidVolume3'] + df['bidVolume4'] + df['bidVolume5']
    df['volumeImbalance1'] = (df['bidVolume1'] - df['askVolume1']) / (df['askVolume1'] + df['bidVolume1'])
    df['volumeImbalanceTotal'] = (df['totalBidVolume'] - df['totalAskVolume']) / (
            df['totalAskVolume'] + df['totalBidVolume'])
    df['volumePerDeal'] = df['volume'] / df['deal']
    df['volumeRatio'] = df['volume'] / total_vol
    df['valuePerDeal'] = df['value'] / df['deal']

    # others
    df['ret_spread'] = (df['bidVolume1'] * df['bidPrice1'] + df['bidVolume2'] * df['bidPrice2'] + df['bidVolume3'] * df[
        'bidPrice3'] \
                        + df['bidVolume4'] * df['bidPrice4'] + df['bidVolume5'] * df['bidPrice5']) / df[
                           'totalAskVolume'] - \
                       (df['askVolume1'] * df['askPrice1'] + df['askVolume2'] * df['askPrice2'] + df['askVolume3'] *
                        df['askPrice3'] + df['askVolume4'] * df['askPrice4'] + df['askVolume5'] * df['askPrice5'])


def _cal_min_features_by_ticks(df):
    def z_score_(arr):
        return arr.mean() / arr.std()

    df_agg = df.groupby('barTime').agg(
        {
            'spread': ['max', 'min', 'mean', 'std'],
            'lastPrice': ['max', 'min', 'mean', 'std'],
            'midPrice': ['max', 'min', 'mean', 'std'],
            'trackError': ['max', 'std'],
            'midSpread': z_score_,
            'deal': ['sum'],
            'volumePerDeal': ['max', 'min', 'mean', 'std'],
            'valuePerDeal': ['max', 'min', 'mean', 'std']
        })
    flatten_columns = ['{0}_{1}'.format(item[0], item[1]) for item in df_agg.columns]
    df_agg.columns = flatten_columns
    df_agg = df_agg.reset_index()
    return df_agg


def _cal_min_features(df_min, window_len=20):
    def _get_skr(arr):
        _mean, _var = arr.mean(), arr.var()
        return ((arr - _mean) ** 3).mean() / (_var ** (2 / 3))

    def _get_kur(arr):
        _mean, _var = arr.mean(), arr.var()
        return ((arr - _mean) ** 4).mean() / (_var ** 2)

    def _get_ac(arr, step):
        if step >= len(arr):
            return np.nan
        # print(arr[:-step], arr[step:])
        _corr = np.corrcoef(arr[:-step], arr[step:])
        # print(_corr)
        return _corr[0][1]

    def _get_rolling_corr(arr1, arr2, window_len):
        ret = [np.nan] * (window_len - 1)
        n_len = len(arr1)
        for i in range(n_len - window_len + 1):
            _r = -1 if i + window_len >= n_len else i + window_len
            _corr = np.corrcoef(arr1[i:_r], arr2[i:_r])
            ret.append(_corr[0][1])
        return ret

    def _get_label_5min(arr):
        arr = list(arr)
        n_len = len(arr)
        ret = []
        for i in range(n_len - 4):
            _r = -1 if i + 5 >= n_len else i + 5
            ret.append(arr[_r] / arr[i] - 1)
        for i in range(4):
            ret.append(np.nan)
        return ret

    # TODO return label by closeprice or vwap??
    # df_min['ret'] = df_min[['closePrice']].rolling(2).apply(lambda x: x[-1] / x[0] - 1) # return label by closePrice
    df_min['ret'] = df_min[['vwap']].rolling(2).apply(lambda x: x[-1] / x[0] - 1)  # return label by vwap
    df_min['retLog'] = df_min[['closePrice']].rolling(2).apply(lambda x: math.log(x[-1]) - math.log(x[0]))
    df_min['retVar'] = df_min[['ret']].rolling(window_len).apply(lambda x: x.var())
    df_min['retSkr'] = df_min[['ret']].rolling(window_len).apply(_get_skr)
    df_min['retKur'] = df_min[['ret']].rolling(window_len).apply(_get_kur)
    df_min['retAc'] = df_min[['ret']].rolling(window_len).apply(_get_ac, kwargs={'step': 5})
    df_min['retBc'] = df_min[['bcClosePrice']].rolling(2).apply(lambda x: x[-1] / x[0] - 1)
    df_min['corrBc'] = _get_rolling_corr(df_min['ret'], df_min['retBc'], window_len=window_len)
    df_min['label'] = ([0.0] + list(df_min['ret']))[1:]
    df_min['label5'] = _get_label_5min(df_min['closePrice'])
    df_min['retTrackBc'] = df_min['ret'] - df_min['retBc']
    df_min['ma5'] = df_min[['closePrice']].rolling(5).mean()
    df_min['ma10'] = df_min[['closePrice']].rolling(10).mean()
    # df_min['ma20'] = df_min[['closePrice']].rolling(20).mean()
    # df_min['maDiff20'] = df_min['ma5'] - df_min['ma20']
    df_min['maDiff10'] = df_min['ma5'] - df_min['ma10']
    df_min['priceAmp'] = (df_min['highPrice'] - df_min['lowPrice']) / df_min['closePrice']
    df_min['sIndicator'] = abs(df_min['ret']) / np.sqrt(df_min['totalVolume'])


def get_features(security_id=u"300634.XSHE", start_date='20191202', end_date='20191206', min_unit="1", tick=False,
                 bc='', win_len=20):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date.replace('-', ''),
                             endDate=end_date.replace('-', ''), isOpen=u"1",
                             field=u"calendarDate", pandas="1")
    t_dates = list(set(df['calendarDate']))
    df_min = DataAPI.MktBarHistDateRangeGet(securityID=security_id, startDate=start_date.replace('-', ''),
                                            endDate=end_date.replace('-', ''),
                                            unit=min_unit, field=u"", pandas="1")
    df_bc_min = DataAPI.MktBarHistDateRangeGet(securityID=bc, startDate=start_date.replace('-', ''),
                                               endDate=end_date.replace('-', ''),
                                               unit=min_unit, field=u"", pandas="1")
    lst = []
    for date in t_dates:
        _df_min = df_min[df_min.dataDate == date]
        _df_bc_min = df_bc_min[df_bc_min.dataDate == date]
        _df = get_features_by_date(security_id=security_id, date=date, min_unit=min_unit, tick=tick, df_min=_df_min,
                                   df_bc_min=_df_bc_min, win_len=win_len)
        lst.append(_df)
    try:
        ret = pd.concat(lst, axis=0)
    except Exception as ex:
        logger.warn('concat the feautres for security id:{0} with error:{1}'.format(security_id, ex))
        ret = None
    return ret


def get_features_by_date(security_id=u"300634.XSHE", date='20191122', min_unit="1", tick=False, df_min=None,
                         df_bc_min=None, win_len=20):
    '''
    Example call:
    -get tick level features:
        get_features_by_date(security_id=u"ticker.mkt", date='yyyymmdd', min_unit="1", tick=True)
    -get min level features:
        get_features_by_date(security_id=u"ticker.mkt", date='yyyymmdd', min_unit="1", tick=False, df_min=xx,df_bc_min=xx)

    '''
    logger.info('Start processing sec id:{0} for  date:{1}'.format(security_id, date))
    df = DataAPI.MktTicksHistOneDayGet(securityID=security_id, date=date.replace('-', ''), startSecOffset="",
                                       endSecOffset="",
                                       field=u"", pandas="1")
    # df_min = DataAPI.SHSZBarHistOneDayGet(tradeDate=date, exchangeCD=exchange_cd, ticker=ticker, unit=min_unit,
    #                                       startTime=u"", endTime=u"", field=u"", pandas="1")

    datatimes = list(df['dataTime'])
    total_vol = list(df['value'])[-1]
    data_min = ['{0}:{1}'.format(item.split(':')[0], item.split(':')[1]) for item in datatimes]
    # tick feature calculation
    df['barTime'] = data_min
    _cal_tick_features(df)
    if tick:
        return df

    df_min['vwap'] = df_min['totalValue'] / df_min['totalVolume']
    df_min['bcClosePrice'] = df_bc_min['closePrice']
    # calculate tick level features

    # calculate min features accumulate from tick level
    df_agg = _cal_min_features_by_ticks(df)

    # calculate min level features
    _cal_min_features(df_min, win_len)

    common_min_lst = set(df_min['barTime']).intersection(set(data_min))
    df_min = df_min[df_min['barTime'].isin(common_min_lst)].sort_values(by='barTime', ascending=True)
    df_agg = df_agg[df_agg['barTime'].isin(common_min_lst)].sort_values(by='barTime', ascending=True)

    df_min = df_min.reset_index()
    df_agg = df_agg.reset_index()
    df_min = pd.concat([df_min, df_agg], axis=1, ignore_index=False)

    df_min['volumePerDeal'] = df_min['totalVolume'] / df_min['deal_sum']
    df_min['valuePerDeal'] = df_min['totalValue'] / df_min['deal_sum']

    df_min = df_min.drop(REMOVE_MIN_COLS, axis=1)
    df_min = df_min.replace(np.inf, np.nan)
    df_min = df_min.replace(-np.inf, np.nan)

    col_before_drop = df_min.columns
    # drop the columns that are all None
    df_min.dropna(axis=1, how='all', inplace=True)
    # df_min.fillna(axis=1, inplace=True, method='pad')
    # drop the rows that contain None
    df_min.dropna(axis=0, how='any', inplace=True)
    col_after_drop = df_min.columns
    if set(col_before_drop) - set(col_after_drop):
        logger.info('Drop the empty columns:{0}'.format(set(col_before_drop) - set(col_after_drop)))

    if not tick:
        return df_min
    else:
        # TODO refactor this session, this is to generate train_x and train_y for time series models(RNN)
        min_vwap = list(df_min['vwap'])
        min_vwap.insert(0, min_vwap[0])
        df_min['ret'] = (df_min['vwap'] / min_vwap[:-1] - 1) * 100
        dict_min_ret = dict(zip(df_min['barTime'], df_min['ret']))
        columns = list(set(df.columns) - set(REMOVE_TICK_COLS))
        df.sort_values(by='dataTime', ascending=True, inplace=True)
        data_min = list(df['barTime'])
        df = df[columns]
        # print(df.head(5))
        # df.fillna(method='ffill', inplace=True)
        # df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
        rows = list(df.values)
        train_x = []
        train_y = []
        _start, _end = 0, 0
        n_row = len(rows)
        total_row = 0
        for idx, val in enumerate(rows):
            hh, mm = data_min[idx].split(':')
            if int(hh) == 14 and int(mm) == 57:
                break
            if int(hh) == 9 and int(mm) <= 30:
                _start = idx
                continue
            if total_row >= 40:
                break

            if idx == n_row - 1:
                if dict_min_ret.get('{0}:{1}'.format(hh, mm)):
                    train_y.append([dict_min_ret.get('{0}:{1}'.format(hh, mm))])
                    train_x.append(list(rows[_start: idx]))
                    total_row += 1
                _start = idx
            else:
                hh_, mm_ = data_min[idx + 1].split(':')
                if int(mm) % 5 == 0 and int(mm_) != int(mm):
                    if dict_min_ret.get('{0}:{1}'.format(hh, mm)):
                        train_y.append([dict_min_ret.get('{0}:{1}'.format(hh, mm))])
                        train_x.append(list(rows[_start: idx + 1]))
                        total_row += 1
                    _start = idx + 1
        logger.info('Start processing sec id:{0} for date:{1}'.format(security_id, date))
        return train_x, train_y, columns


def corr_map(df, fname):
    var_corr = df.corr()
    mask = np.zeros_like(var_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    f, ax = plt.subplots(figsize=(20, 12))
    sns.set(font_scale=1)
    sns.heatmap(var_corr, mask=mask, cmap=cmap, vmax=1, center=0
                , square=True, linewidths=.5, cbar_kws={"shrink": .5}
                , annot=False, annot_kws={'size': 5, 'weight': 'bold', 'color': 'red'})
    # plt.show()
    plt.savefig(get_full_data_path('{0}.jpg'.format(fname)))


def cache_features(start_date='20200302', end_date='20200313', sec_num=30, test_sample=None, saved=True, sample_mode=2,
                   sample_mkt_tickers=['399001']):
    test_sample = test_sample or get_samples(mode=sample_mode, total_num=sec_num, mkt_tickers=sample_mkt_tickers)
    # test_sample = {'399005.XSHE': ['002180.XSHE']}
    # FIXME remove the hardcode of win_len
    for win_len in [10]:
        for k, v in test_sample.items():
            for sec_id in v:
                df = get_features(security_id=sec_id, start_date=start_date, end_date=end_date, min_unit="1",
                                  tick=False,
                                  bc=k, win_len=win_len)
                if not type(df) == pd.DataFrame:
                    logger.info('Exception for calculating features for sec_id:{0} in bc:{1}'.format(sec_id, k))
                    continue
                if not saved:
                    return df
                fname = "{0}_{1}_{2}_{3}".format(sec_id, start_date, end_date, win_len)
                # fname = "{0}_{1}_{2}".format(sec_id, start_date, end_date)
                df.to_csv(get_full_data_path('{0}.csv'.format(fname)), index=False)
                df.drop(['exchangeCD', 'ticker'], axis=1, inplace=True)
                df_corr = df.corr(method='pearson')
                df_corr.to_csv(get_full_data_path('{0}_corr.csv'.format(fname)), index=False)
                label = df_corr['label']
                label5 = df_corr['label5']
                df_corr.drop(['label', 'label5', 'index'], axis=1, inplace=True)
                _df1 = pd.DataFrame({'label': label})
                df_corr1 = pd.concat([_df1, df_corr], axis=1)
                corr_map(df_corr1, fname='{0}_{1}'.format(fname, '1min'))
                _df5 = pd.DataFrame({'label': label5})
                df_corr5 = pd.concat([_df5, df_corr], axis=1)
                corr_map(df_corr5, fname='{0}_{1}'.format(fname, '5min'))


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


def windows_len_search():
    target_features = ['retVar', 'retSkr', 'retKur', 'retAc', 'retBc', 'corrBc', 'ma5', 'ma10', 'ma20']
    for l in [10, 15, 20]:
        ret = dict()
        f_name = '002415.XSHE_20200302_20200313_{0}_corr.csv'.format(l)
        df = pd.read_csv(get_full_data_path(f_name))
        col_names = list(df.columns)

        # labels = df['label']
        labels = df['label5']
        dict_corr = dict(zip(col_names, labels))
        for f in target_features:
            ret.update({f: dict_corr.get(f)})
        print("window len:", l)
        pprint.pprint(ret)


def get_month_start_end_dates(start_date='', end_date=''):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1",
                             field=u"", pandas="1")
    t_dates = list(set(df['calendarDate']))
    t_dates = sorted(t_dates)
    df = df[df.isMonthEnd == 1]
    t_month_end = list(set(df['calendarDate']))
    t_month_end = sorted(t_month_end)
    ret = [t_dates[0]]
    for d in t_month_end:
        try:
            _idx = t_dates.index(d)
            ret.append(d)
            ret.append(t_dates[_idx + 1])
        except Exception as ex:
            pass
    return ret


def main():
    month_start_end_dates = get_month_start_end_dates(start_date='20190104', end_date='20191231')
    idx = 0
    while idx <= 22:
        logger.info("cache features from {0} to {1}".format(month_start_end_dates[idx], month_start_end_dates[idx + 1]))
        cache_features(start_date=month_start_end_dates[idx], end_date=month_start_end_dates[idx + 1], sec_num=10)
        idx += 2

    # get features by date
    # df = get_features_by_date(security_id=u"002415.XSHE", date='20191202', min_unit="1", tick=True)
    # windows_len_search()

    # get features for one month
    # test_sample = {'399005.XSHE': ['002415.XSHE']}
    # df = cache_features(start_date='2019-12-02', end_date='2019-12-02', test_sample=test_sample)
    # print(df.shape)
    # print(df.head(5))
    #

if __name__ == '__main__':
    main()
