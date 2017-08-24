from urllib.request import urlretrieve
import pandas as pd
import os
import time
pd.set_option('display.width', 1000)
pd.set_option('display.column', 1000)


def get_stockPI(stockId,file_package,2):
    target_file=os.path.join(file_package, '{}.csv'.format(stockId))
    urlretrieve(download_url.format(stockId),target_file)
    time.sleep(0.5)
    df=pd.read_csv(target_file,encoding='gbk',delimiter='\t')
    df=df.T
    colnames = list(df.loc['报表日期'])[1:]
    df=df.drop(0, axis=1)
    df=df.iloc[1:30]
    df.columns=colnames
    df.insert(0, '证券代码', stockId)
    df.insert(0, '报告期', df.index)
    return(df)

def get_whole(stock_list,file_package,download_url):
    dfs = []
    i = 0
    for stockId in stock_list:
        stockId = stockId[0:6]
        temp_df = get_stockPI(stockId, file_package, download_url)
        dfs.append(temp_df)
        i = i + 1
        if i % 5 == 0:
            print(i, '/ ', len(stock_list))
    df = pd.concat(dfs, axis=0)
    df = df.drop('19700101', axis=0)
    return(df)





if __name__ == '__main__':
    download_pro = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_ProfitStatement/displaytype/4/stockid/{}/ctrl/all.phtml'
    tempdf = df.drop(list(df[df['营业收入'].isnull()].index))
    tempdf = tempdf.drop('六、每股收益', axis=1)

    download_bs = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_BalanceSheet/displaytype/4/stockid/{}/ctrl/all.phtml'
    download_cfs = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_CashFlow/displaytype/4/stockid/{}/ctrl/all.phtml'

    file_package1 = 'financial_statements'
    file_package2 = 'stock_code'

    f = open('C:\\Users\\bwsstaff\\Desktop\\2768(2016Q1之前上市).txt', 'r')
    stock_2768 = list(f)
    for i in range(len(stock_2768)):
        stock_2768[i]=stock_2768[i][0:9]

    df1 = get_whole(stock_2768,file_package1,download_bs)
    df2 = get_whole(stock_2768, file_package2, download_cfs)
    tempdf1 = df1.drop(list(df1[df1['货币资金'].isnull()].index))
    tempdf2 = df2.drop(list(df2[df2['销售商品、提供劳务收到的现金'].isnull()].index))

    tempdf1.to_csv('C:\\Users\\bwsstaff\\Desktop\\资产负债表数据.csv')
    tempdf2.to_csv('C:\\Users\\bwsstaff\\Desktop\\现金流量表数据.csv')

    tempdf.to_csv('C:\\Users\\bwsstaff\\Desktop\\利润表数据.csv')
    df = pd.read_excel('C:\\Users\\bwsstaff\\Desktop\\利润表数据.xlsx')


t=lambda :setattr(t,'value',3)or 3
t()
t.value
