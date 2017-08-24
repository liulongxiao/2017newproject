from cons import SWZS,DATE_FMT,PICKLE_PATH
import pandas as pd
import requests
from parse_tools import parse_table
import lxml.etree
from datetime import date
from matplotlib import pyplot as plt
import os
from xpinyin import Pinyin
p=Pinyin()


def get_week_data(start_date,end_date):
    file_name=start_date.strftime(DATE_FMT)+end_date.strftime(DATE_FMT)+'-sw'
    if os.path.exists(os.path.join(PICKLE_PATH, file_name)):
        dfs=pd.read_pickle(os.path.join(PICKLE_PATH, file_name))
        return dfs
    p=requests.get(SWZS.format(start_date.strftime(DATE_FMT),end_date.strftime(DATE_FMT)))
    tree=lxml.etree.HTML(p.text)
    data=parse_table(tree.xpath('//table')[0])
    df=pd.DataFrame(data[1:],columns=data[0])
    df.insert(3,'week_time',pd.to_datetime(df['发布日期']).apply(lambda x:str(x.year)+'-'+str(x.week)))
    df['发布日期']=pd.to_datetime(df['发布日期'])
    zhishus=set(df['指数名称'])
    dfs=[]
    for zhishu in zhishus:
        df_inner=df[df['指数名称']==zhishu]
        df_inner=df_inner.sort_values('发布日期')
        df_inner=df_inner.groupby('week_time').apply(lambda x:pd.DataFrame([[x['成交额(亿元)'].astype('float').sum(),float(x['收盘指数'].iat[-1]),x['发布日期'].iat[-1]]],columns=['amount','close','时间']))
        df_inner.insert(0,'指数名称',zhishu)
        df_inner.index=df_inner['时间']
        del df_inner['时间']
        dfs.append(df_inner)
    dfs=pd.concat(dfs)
    dfs.to_pickle(os.path.join(PICKLE_PATH, file_name))
    return dfs


def sw_plot(zhishu,start,end):
    dfs=get_week_data(start,end)
    zhishu_df=dfs[dfs['指数名称']==zhishu]
    zhishu_df=zhishu_df.sort_index()
    fig=plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(zhishu_df.index, zhishu_df['close'].values,'r',label='close')
    ax1.set_ylabel('close')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()  # this is the important function
    ax2.bar(zhishu_df.index,zhishu_df['amount'].values,label='amount',width = 6,alpha=0.5)
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('amount')
    ax2.set_xlabel('time')
    plt.legend(loc='upper right')
    plt.title(p.get_pinyin(zhishu))
    plt.show()



