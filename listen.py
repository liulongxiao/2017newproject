import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime
from datetime import timedelta
import os
import sqlalchemy as scy
from mylog import log


# from WindPy import *

def sql_connect(db):
    try:
        engine = scy.create_engine('mysql+pymysql://root:bwsswb3810@120.77.80.212/{}?charset=utf8'.format(db),
                                   echo=True)
        conn = engine.connect()
    except Exception as e:
        log(e)
    return conn


def get_data(stock_list):
    stock_length = len(stock_list)
    raw_url = 'http://hq.sinajs.cn/list={}'
    try:
        raw_data_text = requests.get(raw_url.format(','.join(stock_list))).text
    except:
        try:
            raw_data_text = requests.get(raw_url.format(','.join(stock_list))).text
        except:
            print(stock_list)
            return pd.DataFrame(columns=['股票名字', '今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价'])
    stocks_raw_data = raw_data_text.split(';')[:stock_length]
    try:
        raw_data = np.array(list(map(lambda x: re.findall('\=\"(.+)\"', x)[0].split(','), stocks_raw_data)))
    except:
        print(stock_list)
        return pd.DataFrame(columns=['股票名字', '今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价'])
    try:
        df = pd.DataFrame(raw_data[:, :6], columns=['股票名字', '今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价'],
                          index=stock_list)
    except Exception as e:
        print(stock_list)
        return pd.DataFrame(columns=['股票名字', '今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价'])
    df.loc[:, ['今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价']] = df.loc[:,
                                                              ['今日开盘价', '昨日收盘价', '当前价格', '今日最高价', '今日最低价']].astype(
        np.float)
    df['当前价格'][df['当前价格'] == 0] = np.nan
    return df


def read_stock(filename):
    stocks = []
    file = open(filename, 'rb')
    for line in file:
        stocks.append(line.decode().replace('\r\n', '').replace('.SZ', '').replace('.SH', '').strip())
    return stocks


def pre_stockId(stockId):
    if stockId[0] == '6':
        return 'sh' + stockId
    else:
        return 'sz' + stockId


def get_total_data(stock_list):
    one_time_length = 30
    data = []
    for i in range(len(stock_list) // one_time_length + 1):
        data.append(get_data(stock_list[i * one_time_length:(i + 1) * one_time_length]))
    return pd.concat(data)


def start_listen():
    time_now = datetime.now().time().strftime('%H %M')
    print('start_listen {}'.format(time_now))
    conn = sql_connect('bainan')

    alert_df = pd.read_sql_query('select stockId,alert_price from stock_alert where alterted=0 order by stockId asc',
                                 conn)
    conn.close()
    alert_df.index = alert_df.stockId
    stock_list = list(map(pre_stockId, alert_df.index))
    this_time = get_total_data(stock_list)['当前价格'].astype(float)
    this_time.index = list(map(lambda x: x[2:], this_time.index))
    this_time - alert_df['alert_price']
    alert_stockId = this_time[(this_time - alert_df['alert_price']) < 0].index
    conn = sql_connect('bainan')
    if len(alert_stockId) != 0:
        if len(alert_stockId) != 1:
            conn.execute('update stock_alert set alterted=1 where stockId in {}'.format(tuple(alert_stockId)))
        else:
            conn.execute('update stock_alert set alterted=1 where stockId="{}"'.format(alert_stockId[0]))
    conn.close()
    if len(alert_stockId) != 0:
        return alert_stockId
    else:
        return None


