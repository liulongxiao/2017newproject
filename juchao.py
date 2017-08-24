import requests
import pandas as pd
from cons import JC_REPORT,JC_DOMAIN
from myutils import get_dir_path
import os




def get_report(stockId):
    p=requests.post(JC_REPORT,data={'stock':stockId,
    'category':'category_ndbg_szsh;category_bndbg_szsh;category_yjdbg_szsh;category_sjdbg_szsh;',
    'pageNum':'1',
    'pageSize':'50',
    'column':'szse_gem',
    'tabName':'fulltext'})
    if p.status_code!=200:
        raise ValueError('status_code {}'.format(p.status_code))
    json_data=p.json()
    df=pd.DataFrame(json_data['announcements'])
    df=df.set_index('announcementTitle')
    return df[['adjunctUrl']]

def download_open_report(url,filename):
    p=requests.get(url)
    path=get_dir_path()
    with open(os.path.join(path,filename+'.pdf'),'wb') as f:
        f.write(p.content)
    os.startfile(os.path.join(path,filename+'.pdf'))

def download_open_report_wrapper(url,filename):
    url=os.path.join(JC_DOMAIN,url)
    download_open_report(url,filename)




