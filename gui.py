from tkinter import *
from tkinter import ttk
from cons import TREE, DATE_FMT, SWZS_MC, PROFIT_COMPONENTS
from level1_data import level1_data
from datetime import date, timedelta, datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import pandas as pd
from listen import start_listen
from tkinter.messagebox import showinfo
from level1_data import data_check
import time
from myutils import get_file_path, get_stock_ids
from sina_data import get_cjzj, get_cjzz, get_lxfl, get_lxsl, get_cjpx, get_zjlr
from tencent_data import get_ddlr, get_ddlc
from sw_data import sw_plot
from fundamental_analyse import plot_revenue_components, plot_profit_indicator
from juchao import download_open_report_wrapper, get_report


class Myframe(Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root

        self.init_components()

    def init_components(self):

        cb = ttk.Combobox(self.root)
        cb['state'] = "readonly"
        cb['values'] = list(TREE.keys())
        cb.bind("<<ComboboxSelected>>", self.show_second_level)
        cb.grid(row=1, column=1)
        self.cb2 = ttk.Combobox(self.root)
        self.cb2.grid(row=2, column=1)
        self.cb3 = ttk.Combobox(self.root)
        self.cb3.grid(row=3, column=1)
        self.cb = cb

        self.show_length_label = Label(text='周期(天)↓')
        self.show_length_label.grid(row=1, column=2)

        self.show_length_var = StringVar()
        self.show_length = Entry(self.root, width=6, textvariable=self.show_length_var)
        self.show_length_var.set('50')
        self.show_length.grid(row=2, column=2)

        self.n_days_var = StringVar()
        self.n_days = Entry(self.root, width=6, textvariable=self.n_days_var)
        self.n_days_var.set('10')
        self.n_days.grid(row=5, column=2)

        self.b1 = Button(self.root, width=22, text='近N天涨幅排名→', command=self.show_n_days)
        self.b1.grid(row=5, column=1)

        self.b2 = Button(self.root, width=22, text='更新数据', command=self.check_data)
        self.b2.grid(row=6, column=1)

        self.b3 = Button(self.root, width=22, text='查看30分钟提醒', command=self.show_30_min)
        self.b3.grid(row=7, column=1)

        self.path = StringVar()
        self.path.set('')

        self.b4 = Button(self.root, width=6, text='浏览', command=self.set_path)
        self.b4.grid(row=7, column=2)

        self.b5 = Button(self.root, width=22, text='股票实时监控', command=self.online_listen)
        self.b5.grid(row=8, column=1)

        Button(self.root, text='申万行业指数↓', width=22).grid(row=9, column=1)
        swzs = ttk.Combobox(self.root)
        swzs['state'] = "readonly"
        swzs['values'] = SWZS_MC
        swzs.bind("<<ComboboxSelected>>", self.swzsplot)
        self.swzs = swzs
        self.swzs.grid(row=10, column=1)

        self.b6 = Button(self.root, width=22, text='股票基本面分析', command=self.fundamental_analyse)
        self.b6.grid(row=11, column=1)

    def fundamental_analyse(self):
        window = Toplevel(self.root)
        e1 = Entry(window, width=22, textvariable=StringVar(window, "输入股票代码"))
        e1.grid(row=0, columnspan=2)

        e2 = Entry(window, width=22, textvariable=StringVar(window, "输入过去对比周期"))
        e2.grid(row=1, columnspan=2)

        report_type = ttk.Combobox(window, width=20)
        report_type['state'] = "readonly"
        report_type['values'] = ['全部', '一季报', '中报', '三季报', '年报']
        report_type.set('年报')
        report_type.grid(row=2, columnspan=2)

        b1 = Button(window, width=22, text='收入构成',
                    command=lambda: self.plot_revenue_components_(e1.get(), e2.get(), report_type.get()))
        b1.grid(row=3, columnspan=2)

        # e3 = Entry(window,width=11, textvariable=StringVar(window, "输入指标名称"))
        # e3.grid(row=4, column=1)

        cb1 = ttk.Combobox(window, width=20)
        cb1['state'] = "readonly"
        cb1['values'] = PROFIT_COMPONENTS
        cb1.bind("<<ComboboxSelected>>",
                 lambda Event: self.plot_profit_indicator_(e1.get(), e2.get(), cb1.get(), report_type.get()))
        cb1.grid(row=4, column=0)

        # b2=Button(window, width=11, text='利润表指标画图',command=lambda:self.plot_profit_indicator_(e1.get(),e2.get(),e3.get(),report_type.get()))
        # b2.grid(row=4,column=0)

        b3 = Button(window, width=22, text='财务报表(pdf)', command=lambda: self.show_report(e1.get()))
        b3.grid(row=5)

    def show_report(self, stockId):
        df = get_report(stockId)

        class my_wrapper:
            def __init__(self, value, index_i, column_j):
                self.value = value
                self.index_i = index_i
                self.column_j = column_j

            def __call__(self):
                showinfo('提示', '报表下载中')
                return download_open_report_wrapper(self.value, self.index_i)

        self.show_table(root, df, my_wrapper)

    def plot_profit_indicator_(self, stockId, limit, indicator, report_type):
        if limit == '输入过去对比周期':
            limit = '4'
        limit = int(limit)
        type = {'全部': None, '一季报': 1, '中报': 2, '三季报': '3', '年报': 4}[report_type]
        plot_profit_indicator(stockId, indicator, limit=limit, type=type)

    def plot_revenue_components_(self, stockId, limit, report_type):
        if limit == '输入过去对比周期':
            limit = '4'
        limit = int(limit)
        type = {'全部': None, '一季报': 1, '中报': 2, '三季报': '3', '年报': 4}[report_type]
        plot_revenue_components(stockId, limit, type)

    def swzsplot(self, Event):
        end = datetime.today()
        start = end - timedelta(days=365)
        zhishu = self.swzs.get()
        sw_plot(zhishu, start, end)

    def online_listen(self):
        window = Toplevel(self.root)
        b1 = Button(window, width=22, text='成交骤减', command=lambda: self.show_table(window, get_cjzj(30)))
        b1.pack()
        b2 = Button(window, width=22, text='成交骤增', command=lambda: self.show_table(window, get_cjzz(30)))
        b2.pack()
        b3 = Button(window, width=22, text='连续放量', command=lambda: self.show_table(window, get_lxfl(30)))
        b3.pack()
        b4 = Button(window, width=22, text='连续缩量', command=lambda: self.show_table(window, get_lxsl(30)))
        b4.pack()
        b5 = Button(window, width=22, text='成交金额排序(实时)', command=lambda: self.show_table(window, get_cjpx(30)))
        b5.pack()
        b6 = Button(window, width=22, text='资金净流入金额排序(实时)', command=lambda: self.show_table(window, get_zjlr(30)))
        b6.pack()
        b7 = Button(window, width=22, text='大单买盘(50万以上,实时)', command=lambda: self.show_table(window, get_ddlr()))
        b7.pack()
        b8 = Button(window, width=22, text='大单卖盘(50万以上,实时)', command=lambda: self.show_table(window, get_ddlc()))
        b8.pack()

    def show_second_level(self, Event):
        cb2 = self.cb2
        cb2['state'] = "readonly"
        cb2['values'] = list(TREE[self.cb.get()].keys())
        cb2.bind("<<ComboboxSelected>>", self.show_third_level)

    def show_third_level(self, Event):
        cb3 = self.cb3
        cb3['state'] = "readonly"
        x = level1_data(self.cb.get(), self.cb2.get(), 'day')
        cb3['values'] = list(x.get_data(date(2017, 7, 31))['商品'])
        cb3.bind("<<ComboboxSelected>>", self.show_detail)

    def show_detail(self, Event):
        x = level1_data(self.cb.get(), self.cb2.get(), 'day')
        x.get_range_data(date.today() - timedelta(int(self.show_length.get())), date.today())
        target_df = x.get_range_data_with_indicator(date.today() - timedelta(int(self.show_length.get())), date.today(),
                                                    self.cb3.get())
        new_window = Toplevel(root)
        f = Figure(figsize=(8, 4), dpi=100)
        a = f.add_subplot(111)
        a.plot(target_df['日期'].apply(lambda x: datetime.strptime(x, DATE_FMT)), target_df['价格'])
        canvas = FigureCanvasTkAgg(f, master=new_window)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, new_window)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

    def show_n_days(self):
        dfs = []
        for first_name in TREE.keys():
            for second_name in TREE[first_name].keys():
                print(first_name, second_name)
                x = level1_data(first_name, second_name, 'day')
                x.get_range_data(date.today() - timedelta(days=int(self.n_days.get())), date.today())
                dfs.append(x.data)
        df = pd.concat(dfs, axis=0)
        df = df.drop_duplicates()
        df = df.sort_values('日期')
        s = df.groupby('商品').apply(lambda x: x['价格'].iloc[-1] / x['价格'].iloc[0])
        s = s.sort_values(ascending=False)
        s_df = pd.DataFrame(s, index=s.index)
        s.to_excel('local_data/latest_n.xlsx')
        window = Toplevel(self.root)
        T = Text(window, height=30, width=50)
        T.pack()
        T.insert(END, s.__str__())
        return

    def show_30_min(self):
        x = start_listen()
        if x is None:
            showinfo('info', '没有股票达到提醒价位')
        else:
            if self.path.get() != '':
                target_stockId = get_stock_ids(self.path.get())
                print(target_stockId)
                alert = set(x).intersection(set(target_stockId))
                if alert.__len__() == 0:
                    showinfo('info', '没有股票达到提醒价位')
                else:
                    showinfo('info', list(alert).__str__() + ':等股票达到提醒价位')
            else:
                showinfo('info', list(x).__str__() + ':等股票达到提醒价位')

    def check_data(self):
        showinfo('info', '数据更新过程中，完成会有提示')
        time.sleep(2)
        data_check()
        showinfo('info', '数据更新完成')

    def show_table(self, theroot, df, button_command=None):
        window = Toplevel(theroot)
        height = df.shape[0] + 1
        width = df.shape[1] + 1
        if button_command:
            for i in range(height):  # Rows
                for j in range(width):
                    if (i == 0) & (j == 0):
                        b = Button(window, width=20, text='')
                        b.grid(row=i, column=j)
                    elif i == 0:
                        b = Button(window, width=20, text=df.columns[j - 1].__str__())
                        b.grid(row=i, column=j)
                    elif j == 0:
                        b = Button(window, width=20, text=df.index[i - 1].__str__())
                        b.grid(row=i, column=j)
                    else:
                        b = Button(window, width=20, text=df.values[i - 1, j - 1].__str__(),
                                   command=button_command(df.values[i - 1, j - 1].__str__(), df.index[i - 1],
                                                          df.columns[j - 1]))
                        b.grid(row=i, column=j)

        else:
            for i in range(height):  # Rows
                for j in range(width):
                    if (i == 0) & (j == 0):
                        b = Entry(window, textvariable=StringVar(window, ""))
                        b.grid(row=i, column=j)
                    elif i == 0:
                        b = Entry(window, text=StringVar(window, df.columns[j - 1].__str__()))
                        b['state'] = 'readonly'
                        b.grid(row=i, column=j)
                    elif j == 0:
                        b = Entry(window, text=StringVar(window, df.index[i - 1].__str__()))
                        b.grid(row=i, column=j)
                    else:
                        b = Entry(window, text=StringVar(window, df.values[i - 1, j - 1].__str__()))
                        b.grid(row=i, column=j)

    def set_path(self):
        self.path.set(get_file_path())
        print(self.path.get())

    def _quit(self):
        '''退出'''
        self.quit()  # 停止 mainloop
        self.destroy()  # 销毁所有部件


root = Tk()
root.title('')
app = Myframe(root)
app.mainloop()
