import pandas as pd
from fundamental_data import get_profit_sheet
from cons import DATE_FMT
from matplotlib import pyplot as plt


def plot_revenue_components(stockId,limit=4,type=None):
    profit_sheet=get_profit_sheet(stockId)
    profit_sheet['quarter']=profit_sheet.index.quarter
    if  type:
        profit_sheet=profit_sheet[profit_sheet['quarter']==type].copy()
    revenue_components=profit_sheet[['营业总收入','营业成本'  ,'销售费用' ,'管理费用']].copy()
    revenue_components['差额']=revenue_components['营业总收入']-revenue_components[['营业成本'  ,'销售费用' ,'管理费用']].sum(axis=1)
    revenue_components=revenue_components.rename(columns={'差额':'others','营业总收入':'revenue','营业成本':'Operation_expense','销售费用':'selling_expenses','管理费用':'dministrative_expenses'})
    revenue_components=revenue_components.head(limit)
    revenue_components=revenue_components.set_index(revenue_components.index.strftime(DATE_FMT))
    revenue_components=revenue_components[['others','Operation_expense','selling_expenses','dministrative_expenses']].copy()
    plot_df=pd.DataFrame(revenue_components.values.T,columns=revenue_components.index,index=revenue_components.columns)
    plot_df.plot.pie(subplots=True,autopct='%1.1f%%')
    plt.show()

def plot_profit_indicator(stockId,indicator,limit=4,type=None):
    profit_sheet = get_profit_sheet(stockId)
    profit_sheet['quarter'] = profit_sheet.index.quarter
    if type:
        profit_sheet = profit_sheet[profit_sheet['quarter'] == type]
    profit_sheet=profit_sheet.sort_index(ascending=True)
    profit_sheet=profit_sheet.set_index(profit_sheet.index.strftime('%Y-%m'))
    profit_sheet[indicator].tail(limit).plot.bar()
    plt.show()

if __name__=='__main__':
    plot_revenue_components('300271',limit=3,type=2)
    plot_profit_indicator('300271','营业成本',limit=5,type=2)