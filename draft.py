#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: order_processing.py
@time: 20-02-27 下午9:03
@desc:
'''

import requests
import pprint
import json
import time
import datetime

opt_url = "http://119.147.211.207:8090/TradeGW/Oper/CreateAccount"
order_url = "http://119.147.211.207:8090/TradeGW/Trade/PlaceXHOrder"
query_url_prefix = "http://119.147.211.207:8090/TradeGW/Query/"
mkt_player_url = "http://119.147.211.207:8090/PlayerCtrl/ChangeMode/"
player_status_url = "http://119.147.211.207:8090/Player/PlayerStatus"


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def crate_account():
    params = {'Username': 'kiki', 'Loginname': 'kiki', 'Telephone': '13760456157', 'Usertype': 0}
    headers = {'Content-type': 'application/json'}
    r = requests.post(opt_url, data=json.dumps(params), headers=headers)
    # pprint.pprint(r.content)
    pprint.pprint(json.loads(r.content.decode('utf-8')))


def place_order(security_id='', order_vol=100, side='1', price=13.0, biz_action=2):
    headers = {'Content-type': 'application/json'}
    data = {'userID': '18', 'assetAccount': 'A000000018', 'market': 'SSE', 'security': '603612.Sh',
            'side': '1', 'bizID': '010', 'bizAction': 4, 'orderQty': 100, 'price': 13.0}
    ticker, market = security_id.split('.')
    data.update({'security': security_id, 'side': side, 'price': price, 'market': market, 'bizAction': biz_action})
    now = datetime.datetime.now()
    # each_vol = int(total_vol / 48 / 100) * 100
    data.update({'orderQty': order_vol})
    while now.hour < 15 and now.minute < 57:
        r = requests.post(order_url, data=json.dumps(data), headers=headers)
        break
        now = datetime.datetime.now()
        time.sleep(300)


def change_mode():
    headers = {'Content-type': 'application/json'}
    data = {'CtrlType': 2, 'PlayerDate': datetime.datetime(2019, 12, 5), 'BeginTime': '093000', 'EndTime': '150000',
            'LevelType': 1, 'SimSpeed': 2}
    r = requests.post(mkt_player_url, data=json.dumps(data, default=json_serial), headers=headers)
    print(r.content)


def player_status():
    # 'ModeID':  '8b49b486-d5a8-4a27-9855-60e2e97e342a',
    headers = {'Content-type': 'application/json'}
    data = {}
    r = requests.post(player_status_url, data=json.dumps(data, default=json_serial), headers=headers)
    pprint.pprint(json.loads(r.content.decode('utf-8')))


def play_mkt():
    headers = {'Content-type': 'application/json'}
    data = {'ModelID': '8b49b486-d5a8-4a27-9855-60e2e97e342a', 'CtrlType': 3,
            'PlayerDate': datetime.datetime(2019, 12, 5), 'BeginTime': '093000', 'EndTime': '150000',
            'LevelType': 1, 'SimSpeed': 2}
    r = requests.post(mkt_player_url, data=json.dumps(data, default=json_serial), headers=headers)
    pprint.pprint(json.loads(r.content.decode('utf-8')))


def query_daily_reports(query_suffix="QueryCurRptList", mode='r', date=None):
    '''
    # QueryCurRptList:日回报明细；QueryCurOrdList:日委托明细；QuerySecurityAssetsList：账户持仓
    '''
    headers = {'Content-type': 'application/json'}
    data = {'userID': '18', 'assetAccount': 'A000000018'}
    f_name = "data/{}_{}.json".format(query_suffix, date or datetime.date.today().strftime("%Y%m%d"))
    if mode == 'w':
        r = requests.post(query_url_prefix + query_suffix, data=json.dumps(data), headers=headers)
        print(query_url_prefix + query_suffix)
        pprint.pprint(data)
        pprint.pprint(headers)
        with open(f_name, 'w') as outfile:
            outfile.write(r.content.decode('utf-8'))
        return json.loads(r.content.decode('utf-8'))
    elif mode == 'r':
        # load the jason file contents
        with open(f_name) as infile:
            contents = infile.read()
            return json.loads(contents)


if __name__ == '__main__':
    # place_order(security_id='603612.SH', order_vol=100, side='1', price=13, biz_action=2)
    # ret = query_daily_reports(query_suffix="QueryCurRptList", mode='w', date='20200320')
    # pprint.pprint(ret)
    # crate_account()
    # change_mode()
    # player_status()
    # play_mkt()
    player_status()
