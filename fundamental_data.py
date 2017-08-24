import pandas as pd
from cons import FUNDAMENTAL_PROFIT_SHEET

def get_profit_sheet(stockId):
    data = pd.read_csv(FUNDAMENTAL_PROFIT_SHEET.format(stockId), encoding='gbk', delimiter='\t')
    df=pd.DataFrame(data.values[1:, 1:].T.astype(float) / 100000000, columns=data.values[1:, 0], index=data.columns[1:])
    df.columns=list(map(lambda x:x.split('、')[-1],df.columns))
    df = df.drop('19700101', axis=0)
    df=df.drop(list(df[df['营业收入'].isnull()].index))
    df=df.set_index(pd.to_datetime(df.index))
    return df


df=get_profit_sheet('300271')