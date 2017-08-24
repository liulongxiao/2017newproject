from cons import CJZJ,CJZJ_PATTERN,CJZZ,CJZZ_PATTERN,LXFL,LXFL_PATTERN,LXSL,LXSL_PATTERN,CJPX,CJPX_PATTERN,ZJLR
import requests
import re
import json
import pandas as pd

def get_cjzj(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(CJZJ.format(number)).text
    data=re.findall(CJZJ_PATTERN, data_raw)
    if data.__len__()==1:
        data_json=json.loads(data[0])
        df = pd.DataFrame(data_json[0]['items'],columns=['股票代码', '股票简称', '收盘价格', '当前成交量',
                                                    '上期成交量', '涨跌', '成交量变化', '成交量变化(百分比)', '日期', 'None'])
        return df[[ '日期','股票代码', '股票简称', '收盘价格', '涨跌', '当前成交量', '上期成交量', '成交量变化(百分比)']]
    else:
        return None

def get_cjzz(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(CJZZ.format(number)).text
    data=re.findall(CJZZ_PATTERN, data_raw)
    if data.__len__()==1:
        data_json=json.loads(data[0])
        df = pd.DataFrame(data_json[0]['items'],columns=['股票代码', '股票简称', '收盘价格', '当前成交量',
                                                    '上期成交量', '涨跌', '成交量变化', '成交量变化(百分比)', '日期', 'None'])
        return df[[ '日期','股票代码', '股票简称', '收盘价格', '涨跌', '当前成交量', '上期成交量', '成交量变化(百分比)']]
    else:
        return None


def get_lxfl(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(LXFL.format(number)).text
    data=re.findall(LXFL_PATTERN, data_raw)
    if data.__len__()==1:
        data_json=json.loads(data[0])
        df = pd.DataFrame(data_json[0]['items'],columns=['股票代码', '股票简称', '收盘价格', '当前成交量',
                                                    '上期成交量', '涨跌', '阶段涨跌','None1', '连续放量天数', '日期', 'None2','None3'])
        return df[[ '日期','股票代码', '股票简称', '收盘价格','涨跌', '阶段涨跌', '连续放量天数']]
    else:
        return None


def get_lxsl(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(LXSL.format(number)).text
    data=re.findall(LXSL_PATTERN, data_raw)
    if data.__len__()==1:
        data_json=json.loads(data[0])
        df = pd.DataFrame(data_json[0]['items'],columns=['股票代码', '股票简称', '收盘价格', '当前成交量',
                                                    '上期成交量', '涨跌', '阶段涨跌','None1', '连续缩量天数', '日期', 'None2','None3'])
        return df[[ '日期','股票代码', '股票简称', '收盘价格','涨跌', '阶段涨跌', '连续缩量天数']]
    else:
        return None


def get_cjpx(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(CJPX.format(number)).text
    data=re.findall(CJPX_PATTERN, data_raw)
    if data.__len__()==1:
        data_json=json.loads(data[0])
        ndarray = pd.DataFrame(data_json[0]['items']).values
        df=pd.DataFrame(ndarray[:,:15],columns=['股票代码','股票代码1','股票名称','最新价','涨跌额',
                                                '涨跌幅','买入','卖出','昨收','今开','最高','最低','成交量(手)','成交额(亿)','时间'])
        df['成交额(亿)']=df['成交额(亿)'].astype(float)/100000000
        df['成交额(亿)']=df['成交额(亿)'].apply(lambda x:round(x,2))
        return df[['股票代码','股票名称','最新价','涨跌幅','昨收','今开','成交额(亿)','时间']]
    else:
        return None


def get_zjlr(number):
    """
    获取成交骤减表格
    :return:
    """
    data_raw=requests.get(ZJLR.format(number))
    symbol,name, trade, changeratio, turnover, amount, inamount, outamount,\
    netamount, ratioamount,r0_in, r0_out, r0_net,r3_in, r3_out, r3_net, \
    r0_ratio, r3_ratio, r0x_ratio= 'symbol,name,trade,changeratio,turnover,amount,inamount,outamount,' \
                                  'netamount,ratioamount,r0_in,r0_out,r0_net,r3_in,r3_out,r3_net,r0_ratio,r3_ratio,r0x_ratio'.split(',')
    data=eval(data_raw.text)
    df=pd.DataFrame(data)
    df=df.rename(columns={'name':'股票简称','symbol':'股票代码','amount':'成交额(亿)','changeratio':'涨跌幅','inamount':'资金流入(亿)',
                          'outamount':'资金流出(亿)','netamount':'资金净流入(亿)','ratioamount':'资金流入比例'})
    df[['成交额(亿)', '资金流入(亿)', '资金流出(亿)', '资金净流入(亿)']]=df[['成交额(亿)','资金流入(亿)','资金流出(亿)','资金净流入(亿)']].astype(float)/100000000
    df[['成交额(亿)', '资金流入(亿)', '资金流出(亿)', '资金净流入(亿)']]=df[['成交额(亿)', '资金流入(亿)', '资金流出(亿)', '资金净流入(亿)']].apply(lambda x:round(x,2))
    return df[['股票代码','股票简称','涨跌幅','成交额(亿)','资金流入(亿)','资金流出(亿)','资金净流入(亿)','资金流入比例']]
