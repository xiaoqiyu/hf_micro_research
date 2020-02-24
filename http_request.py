#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: http_request.py
@time: 20-2-13 上午11:27
@desc:
'''
import requests
import pprint
import json

# url = "http://119.147.211.207:8090//TradeGW/Oper/CreateAccount"
# url = "http://119.147.211.207:8090//TradeGW/Trade/PlaceXHOrder"
url = "http://119.147.211.207:8090//TradeGW/Query/QueryCurRptList"

#create account
data = {'Username': 'kiki', 'Loginname': 'kiki', 'Telephone': '13760456157','Usertype':0}
#place order
# data = {'userID':'34','assetAccount':'010000003402','market':'SZ','security':'300641.SZ',
#         'side':'1','bizID':'010','bizAction':4,'orderQty':100, 'price':13.0}
data = {'userID':'34','assetAccount':'010000003402'}

headers = {'Content-type': 'application/json'}
r = requests.post(url, data=json.dumps(data), headers=headers)
pprint.pprint(r.content)