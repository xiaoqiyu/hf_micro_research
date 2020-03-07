#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: hf_features.py
@time: 19-11-15 下午4:03
@desc:
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import uqer
from uqer import DataAPI
import seaborn as sns
from data_processing.gen_sample import get_samples
from utils.logger import Logger

sns.set()
logger = Logger().get_log()

uqer_client = uqer.Client(token="26356c6121e2766186977ec49253bf1ec4550ee901c983d9a9bff32f59e6a6fb")

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
            _corr = np.corrcoef(arr1[i:i + _r], arr2[i:i + _r])
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

    df_min['ret'] = df_min[['closePrice']].rolling(2).apply(lambda x: x[-1] / x[0] - 1)
    df_min['retLog'] = df_min[['closePrice']].rolling(2).apply(lambda x: math.log(x[-1]) - math.log(x[0]))
    df_min['retVar'] = df_min[['ret']].rolling(window_len).apply(lambda x: x.var())
    df_min['retSkr'] = df_min[['ret']].rolling(window_len).apply(_get_skr)
    df_min['retKur'] = df_min[['ret']].rolling(window_len).apply(_get_kur)
    df_min['retAc'] = df_min[['ret']].rolling(window_len).apply(_get_ac, kwargs={'step': 5})
    df_min['retBc'] = df_min[['bcClosePrice']].rolling(2).apply(lambda x: x[-1] / x[0] - 1)
    df_min['corrBc'] = _get_rolling_corr(df_min['ret'], df_min['retBc'], window_len=20)
    df_min['label'] = ([0.0] + list(df_min['ret']))[1:]
    df_min['label5'] = _get_label_5min(df_min['closePrice'])
    df_min['retTrackBc'] = df_min['ret'] - df_min['retBc']
    df_min['ma5'] = df_min[['closePrice']].rolling(5).mean()
    df_min['ma10'] = df_min[['closePrice']].rolling(10).mean()
    df_min['ma20'] = df_min[['closePrice']].rolling(20).mean()
    df_min['maDiff20'] = df_min['ma5'] - df_min['ma20']
    df_min['maDiff10'] = df_min['ma5'] - df_min['ma10']
    df_min['priceAmp'] = (df_min['highPrice'] - df_min['lowPrice']) / df_min['closePrice']
    df_min['sIndicator'] = abs(df_min['ret']) / np.sqrt(df_min['totalVolume'])


def get_features(security_id=u"300634.XSHE", start_date='20191202', end_date='20191206', min_unit="1", tick=False,
                 bc=''):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date.replace('-', ''),
                             endDate=end_date.replace('-', ''), isOpen=u"1",
                             field=u"calendarDate", pandas="1")
    t_dates = list(df['calendarDate'])
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
                                   df_bc_min=_df_bc_min)
        lst.append(_df)
    try:
        ret = pd.concat(lst, axis=0)
    except Exception as ex:
        logger.warn('concat the feautres for security id:{0} with error:{1}'.format(security_id, ex))
        ret = None
    return ret


def get_features_by_date(security_id=u"300634.XSHE", date='20191122', min_unit="1", tick=False, df_min=None,
                         df_bc_min=None):
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
    _cal_min_features(df_min, 20)

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
        logger.info('Start processing sec id:{0} for bc:{1} and date:{2}'.format(security_id, bc, date))
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
    plt.savefig('{0}.jpg'.format(fname))


def main():
    test_sample = get_samples(mode=0, total_num=30)
    start_date = '20191202'
    end_date = '20191227'
    for k, v in test_sample.items():
        for sec_id in v:
            df = get_features(security_id=sec_id, start_date=start_date, end_date=end_date, min_unit="1", tick=False,
                              bc=k)
            if not type(df) == pd.DataFrame:
                logger.info('Exception for calculating features for sec_id:{0} in bc:{1}'.format(sec_id, k))
                continue
            fname = "data/{0}_{1}_{2}".format(sec_id, start_date, end_date)
            df.to_csv('{0}.csv'.format(fname), index=False)
            df.drop(['exchangeCD', 'ticker'], axis=1, inplace=True)
            df_corr = df.corr(method='pearson')
            df_corr.to_csv('{0}_corr.csv'.format(fname), index=False)
            label_col = df_corr['label']
            df_corr.drop(['label', 'index'], axis=1, inplace=True)
            _df = pd.DataFrame({'label': label_col})
            df_corr = pd.concat([_df, df_corr], axis=1)
            corr_map(df_corr, fname=fname)


if __name__ == '__main__':
    # main()
    df = get_features_by_date(security_id=u"002180.XSHE", date='20191205', min_unit="1", tick=True)
    print(df.shape)