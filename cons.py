import os
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.column', 1000)
TREE = {
    '行业篇':
        {
            '能源榜': '11',
            '化工榜': '14',
            '橡塑榜': '15',
            '纺织榜': '16',
            '有色榜': '12',
            '钢铁榜': '13',
            '建材榜': '17',
            '农副榜': '18',
        },
    '综合榜':
        {
            '大宗商品58榜': 'dz',
            '期现商品榜': 'qx',
            '沪深原材料': 'hs'
        },
    '特色榜':
        {
            '稀土榜': 'xitu',
            '化肥榜': 'huafei',
            '氟化工榜': 'fhg',
            '磷化工榜': 'lhg',
            '溴化工榜': 'xhg',
            '氯碱产业榜': 'ljcy',
            '甲醇产业榜': 'jccy',
            '丙烯产业榜': 'bxcy',
            '苯乙烯产业榜': 'byxcy',
            '乙二醇产业榜': 'y2ccy',
            'PTA产业榜': 'pta',
            '橡胶榜': 'xj',
            '塑料榜': 'sl',
            '资源商品榜': 'zy',
            '商品题材榜': 'sptc',
            '五大钢材榜': 'wdgc'
        }
}
PICKLE_PATH = 'local_data'
DATE_FMT = '%Y-%m-%d'
SHOW_LENGTH = 40

CJZJ='http://money.finance.sina.com.cn/d/api/openapi_proxy.php/?__s=[[%22cjzj%22,%22changes_volume_per%22,1,1,{}]]&callback=getData.block_10'
CJZJ_PATTERN='getData\.block_10\((.+)\)'
CJZZ='http://money.finance.sina.com.cn/d/api/openapi_proxy.php/?__s=[[%22cjzz%22,%22changes_volume_per%22,0,1,{}]]&callback=getData.block_11'
CJZZ_PATTERN='getData\.block_11\((.+)\)'
LXFL='http://money.finance.sina.com.cn/d/api/openapi_proxy.php/?__s=[[%22lxfl%22,%22day_con%22,0,1,{}]]&callback=getData.block_13'
LXFL_PATTERN='getData\.block_13\((.+)\)'
LXSL='http://money.finance.sina.com.cn/d/api/openapi_proxy.php/?__s=[[%22lxsl%22,%22day_con%22,0,1,{}]]&callback=getData.block_14'
LXSL_PATTERN='getData\.block_14\((.+)\)'
CJPX='http://money.finance.sina.com.cn/d/api/openapi_proxy.php/?__s=[[%22hq%22,%22hs_a%22,%22amount%22,0,1,{}]]&callback=FDC_DC.theTableData'
CJPX_PATTERN='FDC_DC.theTableData\((.+)\)'
ZJLR='http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_ssggzj?page=1&num={}&sort=netamount&asc=0&bankuai=&shichang='
DDLR='http://stock.gtimg.cn/data/view/flow.php?t=2'

SWZS="http://www.swsindex.com/excel2.aspx?ctable=swindexhistory&where=  swindexcode in ('801010','801020','801030','801040','801050','801060','801070','801080','801090','801100','801110','801120','801130','801140','801150','801160','801170','801180','801190','801200','801210','801220','801230','801710','801720','801730','801740','801750','801760','801770','801780','801790','801880','801890') and  BargainDate>='{}' and  BargainDate<='{}'"

FUNDAMENTAL_PROFIT_SHEET='http://money.finance.sina.com.cn/corp/go.php/vDOWN_ProfitStatement/displaytype/4/stockid/{}/ctrl/all.phtml'

JC_DOMAIN='http://www.cninfo.com.cn/'
JC_REPORT='http://www.cninfo.com.cn/cninfo-new/announcement/query'

SWZS_MC=['农林牧渔','采掘','化工','黑色金属','有色金属','电子元器件','家用电器','食品饮料','纺织服装','轻工制造','医药生物','公用事业','交通运输',
         '房地产','商业贸易','餐饮旅游','综合','建筑材料','建筑装饰','电气设备','国防军工','计算机','传媒','通信','银行','非银金融','汽车','机械设备']

PROFIT_COMPONENTS=['营业总收入', '营业收入', '营业总成本', '营业成本', '营业税金及附加', '销售费用', '管理费用', '财务费用', '资产减值损失', '公允价值变动收益', '投资收益', '其中:对联营企业和合营企业的投资收益', '汇兑收益', '营业利润', '加:营业外收入', '减：营业外支出', '其中：非流动资产处置损失', '利润总额', '减：所得税费用', '净利润', '归属于母公司所有者的净利润', '少数股东损益', '每股收益', '基本每股收益(元/股)', '稀释每股收益(元/股)', '其他综合收益', '综合收益总额', '归属于母公司所有者的综合收益总额', '归属于少数股东的综合收益总额']


if not os.path.exists(PICKLE_PATH):
    os.mkdir(PICKLE_PATH)
