# -*- coding: utf-8 -*-
# @time      : 2021/4/1 17:22
# @author    : rpyxqi@gmail.com
# @file      : main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def order_samples():
    df_trade = pd.read_csv('mkt/trade1_20210426.csv')
    # 筛选成交状态
    df_trade = df_trade[df_trade.TransactionType != 'C']
    # 计算交易额
    df_trade['Value'] = df_trade['Price'] * df_trade['Volume'] / 10000
    # 筛选股票
    df_trade = df_trade[df_trade.SecID < 100000]
    # bid_df = df_trade[['TradeChannel', 'BidOrder', 'Volume', 'Value']]
    # bid_df.columns = [['TradeChannel', 'Order', 'Volume', 'Value']]
    ask_df = df_trade[['TradeChannel', 'AskOrder', 'Volume', 'Value']]
    ask_df.columns = ['TradeChannel', 'Order', 'Volume', 'Value']
    _df = df_trade[['TradeChannel', 'BidOrder', 'Volume', 'Value']]
    _df.columns = ['TradeChannel', 'Order', 'Volume', 'Value']
    bid_df = _df.groupby(['TradeChannel', 'Order']).apply(sum)
    ask_df = ask_df.groupby(['TradeChannel', 'Order']).apply(sum)

    del df_trade

    df_vwap = pd.concat([ask_df, bid_df])

    df_vwap = df_vwap.sample(frac=0.5, replace=False, random_state=None, axis=0)

    order_lst = list(set(df_vwap.index))

    df_quote = pd.read_csv('mkt/order_20210426.csv')[
        ['SecID', 'Market', 'TradeChannel', 'Order', 'Side', 'Price', 'Volume', 'OrderType', 'OrigTimeStamp']]
    df_quote = df_quote[df_quote.OrderType == '2']
    df_quote = df_quote.set_index(['TradeChannel', 'Order'])

    df_quote = df_quote.loc[order_lst].dropna(axis=0, how='all')
    order_lst = list(set(df_quote.index))

    df_vwap = df_vwap.loc[order_lst].dropna(axis=0, how='all')

    df_vwap['vwap'] = df_vwap['Value'] / df_vwap['Volume']
    df_final = df_quote.join(df_vwap, lsuffix='_l', rsuffix='_r')
    df_final['Price'] = df_final['Price'] / 10000
    df_final['PriceErr'] = abs(df_final['Price'] / df_final['vwap'] - 1)
    df_final['VolumeErr'] = abs(df_final['Volume_l'] / df_final['Volume_r'] - 1)
    df_final.to_csv('mkt/final50_20210426.csv')


def gen_order_samples():
    df_trade = pd.read_csv('trade_20210317.csv').head(1000)
    # 筛选成交状态
    df_trade = df_trade[df_trade.TransactionType != 'C']
    # 计算交易额
    df_trade['Value'] = df_trade['Price'] * df_trade['Volume'] / 10000
    # 筛选股票
    df_trade = df_trade[df_trade.SecID < 100000]

    # df_trade['AskOrder'] = df_trade['AskOrder'].astype(str)
    # df_trade['BidOrder'] = df_trade['BidOrder'].astype(str)
    #  = df_trade['TradeChannel'].astype(str)

    df_trade['AskOrder'] = [str(int(item)) for item in df_trade['AskOrder']]
    df_trade['BidOrder'] = [str(int(item)) for item in df_trade['BidOrder']]
    df_trade['TradeChannel'] = [str(int(item)) for item in df_trade['TradeChannel']]

    ask_df = df_trade[['AskOrder', 'Value', 'Volume', 'TradeChannel']]
    bid_df = df_trade[['BidOrder', 'Value', 'Volume', 'TradeChannel']]

    _tmp_ask_order = list(ask_df['TradeChannel'])
    _tmp_ask_tc = list(ask_df['AskOrder'])
    # _check_lst = []
    # _len = len(_tmp_ask_order)
    # for idx in range(_len):
    #     try:
    #         _check_lst.append('{0}_{1}'.format(_tmp_ask_order[idx], _tmp_ask_tc[idx]))
    #     except Exception as ex:
    #         print(ex)
    # from collections import Counter
    # _check_dict = dict(Counter(_check_lst))
    # print([key for key, value in _check_dict.items() if value > 1])
    # _check_lst = ['{0}_{1}'.format(item, _tmp_ask_tc[idx]) for idx,item in enumerate(_tmp_ask_order)]

    ask_df['Order'] = ask_df['TradeChannel'] + ask_df['AskOrder']
    bid_df['Order'] = bid_df['TradeChannel'] + bid_df['BidOrder']

    _tmp_ask_order = ask_df['Order']
    _tmp_bid_order = bid_df['Order']

    _tmp_ask_order_int = []
    _tmp_bid_order_int = []
    for item in _tmp_ask_order:
        try:
            _tmp_ask_order_int.append(int(item))
        except Exception as ex:
            print(ex)

    for item in _tmp_bid_order:
        try:
            _tmp_bid_order_int.append(int(item))
        except Exception as ex:
            print(ex)
    ask_df['NewOrder'] = _tmp_ask_order_int
    bid_df['NewOrder'] = _tmp_bid_order_int
    print('before groupby')
    print(bid_df[bid_df.Order == 20114504780])
    ask_df = ask_df.groupby('NewOrder').apply(sum)[['NewOrder', 'Value', 'Volume']]
    bid_df = bid_df.groupby('NewOrder').apply(sum)[['NewOrder', 'Value', 'Volume']]
    # bid/ask in trade order is int
    print('after groupby bid')
    print(bid_df[bid_df.NewOrder == 20114504780])
    print('after groupby ask')
    print(ask_df[ask_df.NewOrder == 20114504780])
    df_vwap = pd.concat([ask_df, bid_df])
    df_vwap.columns = ['NewOrder', 'DealValue', 'DealVolume']
    print('after concat')
    print(df_vwap[df_vwap.NewOrder == 20114504780])
    df_vwap = df_vwap.sample(frac=0.8, replace=False, random_state=None, axis=0)

    df_quote = pd.read_csv('order_20210317.csv')
    df_quote = df_quote[df_quote.OrderType == '2']

    # 从交易来的order set
    _order_set = set(df_vwap['NewOrder'])
    _order_set_str = [str(int(item)) for item in _order_set]

    # df_quote['TradeChannel'] = df_quote['TradeChannel'].astype(str)
    # df_quote['Order'] = df_quote['Order'].astype(str)
    # df_quote['Order'] = df_quote['TradeChannel'] + df_quote['Order']

    df_quote['TradeChannel'] = [str(int(item)) for item in df_quote['TradeChannel']]
    df_quote['Order'] = [str(int(item)) for item in df_quote['Order']]
    df_quote['NewOrder'] = df_quote['TradeChannel'] + df_quote['Order']

    print('in quote')
    print(df_quote[df_quote.Order == '20114504780'])
    # 从委托中筛选出交易的order,可能这个set减少了，因为只保留限价委托
    df_quote_sampe = df_quote[df_quote['NewOrder'].isin(_order_set_str)][[
        'SecID', 'Market', 'NewOrder', 'Side', 'Price', 'Volume', 'OrderType', 'OrigTimeStamp']]

    # df_quote_sampe.to_csv(
    #     'mkt/df_quote_sample.csv', index=False)

    # 最终的order set, str 类型
    _order_lst = set(df_quote_sampe['NewOrder'])

    #
    df_vwap['VWAP'] = df_vwap['DealValue'] / df_vwap['DealVolume']
    # df_vwap = df_vwap[df_vwap['Order'].isin(_order_lst)]
    # df_vwap.to_csv('mkt/vwap.csv', index=False)

    # 交易数据Order变成str
    _tmp_order = [str(int(item)) for item in df_vwap['NewOrder']]
    df_vwap['NewOrder'] = _tmp_order

    # 对交易根据委托再进一步筛选，只保留限价委托
    df_vwap = df_vwap[df_vwap["NewOrder"].isin(_order_lst)]

    # df_vwap = df_vwap.set_index('Order')
    # df_quote_sampe = df_quote_sampe.set_index('Order')

    df_quote_sampe['NewOrder'] = df_quote_sampe['NewOrder'].astype(str)
    df_vwap['NewOrder'] = df_vwap['NewOrder'].astype(int)

    _tmp_lst = []
    for item in list(df_quote_sampe['NewOrder']):
        try:
            _tmp_lst.append(int(item))
        except Exception as ex:
            print(ex)
    df_quote_sampe['NewOrder'] = _tmp_lst

    _tmp_lst = []
    for item in list(df_vwap['NewOrder']):
        try:
            _tmp_lst.append(int(item))
        except Exception as ex:
            print(ex)
    df_vwap['NewOrder'] = _tmp_lst

    df_final = df_quote_sampe.join(df_vwap, on='NewOrder', lsuffix='_l', rsuffix='_r')
    price_lst = [item / 10000 for item in df_final['Price']]
    df_final['OrderPrice'] = price_lst
    # df_final['Order'] = list(df_final.index)
    df_final['err'] = abs(df_final['OrderPrice'] / df_final['VWAP'] - 1)
    df_final.to_csv('mkt/final80.csv', index=False)


def process_from_raw():
    with open("quote_data_02_2021-4-26.txt") as f:
        lines = f.readlines()
        # print(f.readline())
        lst = []
        for item in lines:
            try:
                lst.append(item.strip().strip('[').strip(']').split(']['))
            except Exception as ex:
                print(ex)
        del lines
        # 20404 逐笔委托，20405逐笔成交
        quote_df = pd.DataFrame([item for item in lst if item[2] == '20405'])
        # quote_df.to_csv("quoteorder.csv", index=False)
        # 委托字段
        # quote_df.columns = ["SerialNo", "TimeStamp", "MsgHead", "MsgID", "SecID", "Market", "OrigTime", "TradeChannel",
        #                     "Order", "Side", "Price", "Volume", "OrderType"]
        # 成交字段
        # quote_df.columns = ["SerialNo", "TimeStamp", "MsgHead", "MsgID", "SecID", "Market", "OrigTime", "TradeChannel",
        #                     "TransactionIndex",
        #                     "Price", "Volume", "BSFlag", "TransactionType", "AskOrder", "BidOrder"]
        # quote_df['TimeStamp'].astype('str')
        # quote_df['OrigTime'].astype('str')
        # quote_df['OrigTimeStamp'] = [item[-9:] for item in quote_df['OrigTime']]
        quote_df['OrigTimeStamp'] = [item[-9:] for item in quote_df.iloc[:, 6]]
        quote_df = quote_df[quote_df.OrigTimeStamp >= '0932000']
        quote_df = quote_df[quote_df.OrigTimeStamp <= '1457000']
        quote_df.to_csv('mkt/trade_20210426.csv', index=False)
        # tmp_df = quote_df[quote_df.SecID == "300124"]
        # tmp_df.to_csv("trade_300214.csv", index=False)
        # print(len(lst))


def process_sample_order():
    df = pd.read_csv('mkt/final50_20210426.csv')
    df = df[df.VolumeErr <= 1]
    df = df[df.OrigTimeStamp >= 93200000]
    df = df.sort_values(by=['TradeChannel', 'Order'])
    trade_df = df[
        ['TradeChannel', 'Order', 'SecID', 'Market', 'Side', 'Price', 'Volume_l', 'OrderType', 'OrigTimeStamp',
         'Volume_r', 'Value', 'vwap']]
    trade_df.columns = ['TradeChannel', 'Order', 'SecID', 'Market', 'Side', 'Price', 'Volume', 'OrderType', 'TimeStamp',
                        'DealVolume', 'DealValue', 'VWAP']
    trade_df = trade_df.sample(frac=0.5, replace=False, random_state=None, axis=0)
    trade_df.to_csv('mkt/trade_sample_20210426.csv', index=False)
    order_df = trade_df[
        ['TradeChannel', 'Order', 'SecID', 'Market', 'Side', 'Price', 'Volume', 'OrderType', 'TimeStamp']]
    order_df.to_csv('mkt/order_sample_20210426.csv', index=False)
    # df.to_csv('mkt/full50_20210317.csv', index=False)


def analysis_results():
    df = pd.read_csv('trade_samplev0_20210426.csv')
    df_timeline = pd.read_csv('simulation_trade_20210426.csv')
    df['vwap_timeline'] = df_timeline['VWAP']
    df['vol_track1'] = df['Volume'] - df['DealVolume']
    df = df[df.vol_track1 == 0]
    print(df.shape)

    df = df[df.vwap_timeline >0]
    print(df.shape)
    # lst = np.random.normal(0, 0.0005, df.shape[0])
    price_order = list(df['Price'])
    price_trade = list(df['VWAP'])
    # price_test_lst = price_order * (1 + lst) //simulate with noice

    price_test_lst = list(df['vwap_timeline'])
    _track_err_lst = get_track_err(price_test_lst, price_trade)
    ret_sts = get_sts(_track_err_lst)
    print(ret_sts)
    # plt.plot(price_order_err)
    # plt.savefig('order.png')


def get_track_err(lst1, lst2):
    return [abs(item / lst2[idx] - 1) for idx, item in enumerate(lst1)]


def get_sts(inputs=[]):
    s50 = np.quantile(inputs, 0.5)
    s75 = np.quantile(inputs, 0.75)
    s90 = np.quantile(inputs, 0.9)
    s95 = np.quantile(inputs, 0.95)
    s98 = np.quantile(inputs, 0.98)
    _max = max(inputs)
    _avg = sum(inputs) / len(inputs)
    _std = np.array(inputs).std()
    return [_max, _avg, _std, s50, s75, s90, s95, s98]


if __name__ == "__main__":
    # df = pd.read_csv('mkt/trade1_20210426.csv')
    # df.columns = ["SerialNo", "TimeStamp", "MsgHead", "MsgID", "SecID", "Market", "OrigTime", "TradeChannel", "TransactionIndex",
    #                          "Price", "Volume", "BSFlag", "TransactionType", "AskOrder", "BidOrder", "unknown","OrigTimeStamp"]
    # df = df.drop(columns=["unknown"])
    # df.to_csv('mkt/trade1_20210426.csv', index=False)
    # print(df.shape)
    # process_from_raw()
    # order_samples()
    # process_sample_order()
    analysis_results()
