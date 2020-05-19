#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: lstm_predictions.py
@time: 2020/4/10 10:49
@desc:
'''

from data_processing.hf_features import cache_features
from data_processing.hf_features import get_month_start_end_dates
from data_processing.gen_sample import get_samples
from model_processing.lstm_clf_model import train_lstm
from model_processing.lstm_clf_model import predict_with_lstm
from utils.logger import Logger
import pandas as pd
import pprint

SAMPLE_MODE = 2
SAMPLE_NUM = 1
TEST_DATE = '2019-12-02'
MKT_TICKERS = ['399001']

logger = Logger().get_log()


def main():
    # get test samples for train and test
    test_samples = get_samples(mode=SAMPLE_MODE, total_num=SAMPLE_NUM, mkt_tickers=MKT_TICKERS)

    # #calculate features by more than one month period
    # month_start_end_dates = get_month_start_end_dates(start_date='20190104', end_date='20191231')
    # idx = 0
    # while idx <= 22:
    #     logger.info("cache features from {0} to {1}".format(month_start_end_dates[idx], month_start_end_dates[idx + 1]))
    #     cache_features(start_date=month_start_end_dates[idx], end_date=month_start_end_dates[idx + 1],
    #                    test_sample=test_samples)
    #     idx += 2

    # #calculate the features by the period within one month
    # cache_features(start_date='20190104', end_date='20190131', test_sample=

    all_sec_ids = []
    for k, v in test_samples.items():
        all_sec_ids.extend(v)

    # train for sec_id for TEST_DATE
    # for sec_id in all_sec_ids:
    #     train_lstm(test_date=TEST_DATE, security_id=sec_id)

    labels, accuracy = predict_with_lstm(date=TEST_DATE, predict_sample=test_samples)
    cols = list(labels.keys())
    indexes = list()
    from collections import defaultdict
    vals = defaultdict(list)
    for sec_id, pred in labels.items():
        indexes = list(pred.keys())
        vals[sec_id] = list(pred.values())
    df = pd.DataFrame(vals, index=indexes, columns=cols)
    df.to_csv("pred_results_{0}.csv".format(TEST_DATE))
    logger.info('accuracy results is:{0}'.format(accuracy))
    pprint.pprint(df)
    pprint.pprint(accuracy)


if __name__ == "__main__":
    main()
