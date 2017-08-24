import pandas as pd
from cons import TREE, PICKLE_PATH, DATE_FMT
import os
import re
from urllib.parse import urlsplit
from http.client import UnknownProtocol
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt, retry_if_exception
from urllib.error import URLError


def get_stock_ids(path):
    with open(path, 'r') as f:
        stockIds = f.read()
    stocks = re.findall('\d{6}', stockIds)
    return stocks


def get_dir_path():
    from win32com.shell import shell
    file_info = shell.SHBrowseForFolder()
    if file_info[0] is None:
        raise ValueError("目标文件夹不存在或者存在其他错误")
    return shell.SHGetPathFromIDList(file_info[0]).decode()


def get_file_path():
    from win32gui import GetOpenFileNameW
    file_info = GetOpenFileNameW(Filter='*.txt\0')
    if file_info[0] is None:
        raise ValueError("目标文件夹不存在或者存在其他错误")
    return file_info[0]


def generate_url(first_name, second_name, ktype):
    if first_name == '行业篇':
        url = 'http://top.100ppi.com/zdb/detail-{}-{}-{}-1.html'.format(ktype, '{}', TREE[first_name][second_name])
    elif first_name == '综合榜':
        url = 'http://top.100ppi.com/{}/detail-{}-{}-1-1.html'.format(TREE[first_name][second_name], ktype, '{}')
    elif first_name == '特色榜':
        url = 'http://top.100ppi.com/{}/detail-{}-{}-1-1.html'.format(TREE[first_name][second_name], ktype, '{}')
    else:
        raise NotImplemented('该一级名称没有实现 {}'.format(first_name))
    return url


@retry(retry=retry_if_exception_type(URLError), wait=wait_fixed(5), stop=stop_after_attempt(3))
@retry(retry=retry_if_exception_type(UnknownProtocol), wait=wait_fixed(5), stop=stop_after_attempt(3))
def get_table_from_url(url, date):
    df = pd.read_csv(url, encoding='gbk', delimiter='\t', index_col=False)
    if df.empty:
        print(url)
        return pd.DataFrame(columns=['商品', '行业', '价格'])
    df = pd.DataFrame(df.values[:, :3], columns=['商品', '行业', '价格'])
    df.insert(1, '日期', date)
    df['价格'] = df['价格'].astype(float)
    return df


def get_table(first_name, second_name, ktype, date):
    format_date = date.strftime('%Y-%m%d')
    url = generate_url(first_name, second_name, ktype).format(format_date)
    file_name = urlsplit(url).path.replace('/', '-').replace('.html', '')
    if os.path.exists(os.path.join(PICKLE_PATH, file_name)):
        df = pd.read_pickle(os.path.join(PICKLE_PATH, file_name))
        return df
    else:
        print('getting {}'.format(format_date))
        df = get_table_from_url(url, format_date)
        df.to_pickle(os.path.join(PICKLE_PATH, file_name))
        return df


def get_table_check(first_name, second_name, ktype, date):
    format_date = date.strftime('%Y-%m%d')
    url = generate_url(first_name, second_name, ktype).format(format_date)
    file_name = urlsplit(url).path.replace('/', '-').replace('.html', '')
    if os.path.exists(os.path.join(PICKLE_PATH, file_name)):
        df = pd.read_pickle(os.path.join(PICKLE_PATH, file_name))
        if df.empty:
            if date.isoweekday() not in (6, 7):
                print('getting {}'.format(format_date))
                df = get_table_from_url(url, format_date)
                df.to_pickle(os.path.join(PICKLE_PATH, file_name))
                return df
        return df
    else:
        print('getting {}'.format(format_date))
        df = get_table_from_url(url, format_date)
        df.to_pickle(os.path.join(PICKLE_PATH, file_name))
        return df
