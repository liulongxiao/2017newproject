import requests
from cons import DDLR
import pandas as pd
def get_ddlr():
    raw_data=requests.get(DDLR).content.decode('gbk')
    raw_data=raw_data.replace('\'','').replace('\"','')
    raw_data=raw_data.split(';')[0].split('=')[1].split('^')
    raw_data=list(map(lambda x:x.split('~'),raw_data))
    df=pd.DataFrame(raw_data,columns=['行业代码','行业名称','None1','None2','大单买入金额(亿)','None4','大单买入占比'])
    df['大单买入金额(亿)']=(df['大单买入金额(亿)'].astype(float)/10000).apply(lambda x:round(x,2))
    df['大单买入占比']=df['大单买入占比'].apply(lambda x:abs(float(x)))
    df=df[df['大单买入金额(亿)']>0]
    return df[['行业名称','大单买入金额(亿)','大单买入占比']]

def get_ddlc():
    raw_data = requests.get(DDLR).content.decode('gbk')
    raw_data = raw_data.replace('\'', '').replace('\"', '')
    raw_data = raw_data.split(';')[0].split('=')[1].split('^')
    raw_data = list(map(lambda x: x.split('~'), raw_data))
    df = pd.DataFrame(raw_data, columns=['行业代码', '行业名称', 'None1', 'None2', '大单卖出金额(亿)', 'None4', '大单卖出占比'])
    df['大单卖出金额(亿)'] = (df['大单卖出金额(亿)'].astype(float) / 10000).apply(lambda x: round(x, 2))
    df['大单卖出占比'] = df['大单卖出占比'].apply(lambda x: abs(float(x)))
    df=df[df['大单卖出金额(亿)']<0]
    df['大单卖出金额(亿)'] = df['大单卖出金额(亿)'].apply(lambda x:abs(x))
    df=df.sort_values('大单卖出金额(亿)',ascending=False)
    df.index=range(df.shape[0])
    return df[['行业名称', '大单卖出金额(亿)', '大单卖出占比']]
