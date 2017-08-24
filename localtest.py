import sqlalchemy as scy
from mylog import log
import pandas as pd
import tushare as ts
# from WindPy import *

def sql_connect(db):
    try:
        engine = scy.create_engine('mysql+pymysql://root:bwsswb3810@120.77.80.212/{}?charset=utf8'.format(db),
                                   echo=True)
        conn = engine.connect()
    except Exception as e:
        log(e)
    return conn

conn=sql_connect('stock_data')
stockIds=set(pd.read_sql_query('select stockId from stock_basic',conn).stockId)



def get_liulei(stockId):
    df=ts.get_k_data(stockId,start='2016-08-15')
    if df.empty:
        return None
    themax=df.high.astype(float).max()
    themin=df.low.astype(float).min()
    current=df.close.astype(float).iat[-1]
    upward=current/themin-1
    downward=1-current/themax
    return [stockId,themax,themin,current,upward,downward]

df_list=[]
stockIds=list(stockIds)
for stockId in stockIds[:10]:
    data=get_liulei(stockId)
    print(stockId)
    if data is None:
        continue
    else:
        df_list.append(data)
df=pd.DataFrame(df_list,columns=['股票代码','最高价','最低价','当前价格','最低价涨幅','最高价跌幅'])
df.to_excel('leige.xlsx')



