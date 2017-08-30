import pickle
from cons import STOCKID_PICKLE,HANGYE_PICKLE,HANGYE_KEYS,INDEX,INDICATORS,TRAIN_DATA_PATH
import tushare as ts
import pandas as pd
import numpy as np
import os
import random
with open(INDEX,'rb') as f:
    index=pickle.load(f)


def get_total_stockid():
    with open(STOCKID_PICKLE,'rb') as f:
        data=pickle.load(f)
    return data

class get_stock_hangye:
    with open(HANGYE_PICKLE,'rb') as f:
        stock_hangye=pickle.load(f)
    total_stockId=get_total_stockid()
    @staticmethod
    def get_stock_hangye(stockId):
        return get_stock_hangye.stock_hangye.hangye.at[stockId]
    @staticmethod
    def get_hangte_stock(hangye):
        return set(get_stock_hangye.stock_hangye[get_stock_hangye.stock_hangye.hangye==hangye].index).intersection(get_stock_hangye.total_stockId)

def get_hangye_keys():
    with open(HANGYE_KEYS, 'rb') as f:
        hangye_keys = pickle.load(f)
    return hangye_keys

def get_hanye_total_data(hanye):
    stockIds=get_stock_hangye.get_hangte_stock(hanye)
    dfs=[]
    for stockId in stockIds:
        df=ts.get_k_data(stockId)
        if not df.empty:
            df[['open','close','low','high','volume']]=df[['open','close','low','high','volume']].astype(float)
            df['thereturn']=df['close'].rolling(window=2).apply(lambda x:x[1]/x[0]-1).shift(-1).fillna(0)
            df.index=df.date
            df=df.reindex(index)
            df.code = df.code[~df.code.isnull()][0]
            df.date=df.index
            df.close=df.close.fillna(method='ffill').fillna(method='bfill')
            df.volume=df.volume.fillna(0)
            df.thereturn=df.thereturn.fillna(0)
            df.open[df.open.isnull()]=df.close[df.open.isnull()]
            df.high[df.high.isnull()] = df.close[df.high.isnull()]
            df.low[df.low.isnull()] = df.close[df.low.isnull()]
            df=df[df.date>'2015-02-01']
            dfs.append(df)
    return pd.concat(dfs,axis=0)

def df2ndarray(df,indicator):
    df=df.copy()
    df.code=df.code.astype(str)
    stockIds=sorted(list(set(df.code)))
    dates=sorted(list(set(df.date)))
    result=np.empty((len(dates),len(stockIds)))
    a = [df.index, df.code]
    tuples = list(zip(*a))
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df.index=multiindex

    for stockIdx,stockId in enumerate(stockIds):
        for dateIdx,date in enumerate(dates):
            try:
                result[dateIdx,stockIdx]=df.at[(date,stockId),indicator]
            except:
                print(stockId,date,indicator)
        print(stockIdx)
    return result


def download_data(hangye):
    df=get_hanye_total_data(hangye)
    for indicator in INDICATORS:
        result=df2ndarray(df,indicator)
        with open(os.path.join(TRAIN_DATA_PATH,hangye+indicator),'wb') as f:
            pickle.dump(result,f)
            print('{}:{}'.format(hangye,indicator))

def retrieve_data(hangye):
    data={}
    for indicator in INDICATORS:
        with open(os.path.join(TRAIN_DATA_PATH,hangye+indicator),'rb') as f:
            inner_data=pickle.load(f)
            inner_data_shape=inner_data.shape
            inner_data=inner_data.reshape(*inner_data_shape,1)
            data[indicator]=inner_data
    return data


def retrieve_return(hangye):
    with open(os.path.join(TRAIN_DATA_PATH, hangye + 'thereturn'), 'rb') as f:
        inner_data = pickle.load(f)
    return inner_data

def dict2ndarray(data):
    return np.concatenate([data['open'],data['close'],data['high'],data['low'],data['volume']],axis=-1)


def generate_data(data,labels_data,batch_size,num_of_stages):
    data=data.copy()
    labels_data=labels_data.copy()
    data_lengh=data.shape[0]
    results=[]
    labels=[]
    for i in range(batch_size):
        start_point=np.random.randint(0,data_lengh-num_of_stages)
        end_point=start_point+num_of_stages
        result=data[start_point:end_point]
        label=labels_data[start_point:end_point]
        result_shape=result.shape
        label_shape=label.shape
        results.append(result.reshape(1,*result_shape))
        labels.append(label.reshape(1,*label_shape))
    return np.concatenate(results,axis=0),np.concatenate(labels,axis=0)


if __name__=="__main__":
    download_data('医药生物')
    data=retrieve_data('医药生物')
    thereturn=retrieve_return('医药生物')
    X,y=generate_data(data,thereturn,50,20)
    X.shape
    y.shape