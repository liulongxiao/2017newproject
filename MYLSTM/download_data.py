from WindPy import *
w.start()
def wind_code_process(stockId):
    if stockId[0]==6:
        return stockId+'.SH'
    else:
        return stockId+'.SZ'

def get_5_minute_data(stockId,start_time,end_time):
    wind_code=wind_code_process(stockId)
    data=w.wsi(wind_code, "open,high,low,close,volume,amt",start_time, end_time, "BarSize=5")
    if data.ErrorCode!=0:
        pass






