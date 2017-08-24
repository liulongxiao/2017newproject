import pandas as pd
from cons import TREE, DATE_FMT
from myutils import get_table, get_table_check


class level1_data:
    def __init__(self, first_name, second_name, ktype):
        self.first_name = first_name
        self.second_name = second_name
        self.ktype = ktype
        self.parsed_date = set()

    def get_data(self, target_date):
        if target_date not in self.parsed_date:
            df = get_table(self.first_name, self.second_name, self.ktype, target_date)
            if hasattr(self, 'data'):
                self.data = pd.concat([self.data, df], axis=0)
            else:
                self.data = df
            return df

    def get_data_check(self, target_date):
        if target_date not in self.parsed_date:
            df = get_table_check(self.first_name, self.second_name, self.ktype, target_date)
            if hasattr(self, 'data'):
                self.data = pd.concat([self.data, df], axis=0)
            else:
                self.data = df
            return df

    def get_range_data(self, start_date, end_date):
        datelist = pd.date_range(start_date, end_date)
        for date in datelist:
            self.get_data(date)

    def get_range_data_check(self, start_date, end_date):
        datelist = pd.date_range(start_date, end_date)
        for date in datelist:
            self.get_data_check(date)

    def get_range_data_with_indicator(self, start_date, end_date, indicator):
        data = self.data.copy()
        data.index = range(data.shape[0])
        data = data[(data['日期'] >= start_date.strftime(DATE_FMT)) & (data['日期'] <= end_date.strftime(DATE_FMT)) & (
            data['商品'] == indicator)]
        data = data.sort_values('日期')
        return data


def data_check():
    from datetime import date
    from datetime import timedelta
    for first_name in TREE.keys():
        for second_name in TREE[first_name].keys():
            print(first_name, second_name)
            x = level1_data(first_name, second_name, 'day')
            x.get_range_data_check(date.today() - timedelta(days=10), date.today() - timedelta(days=2))
