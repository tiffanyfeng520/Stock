'''
Author: Yatong Feng
Email: yf2563@cumc.columbia.edu
date: 7/21/20 15:49
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
sns.set()
import tushare as ts
token = '0588c520ede362cfacfaf8cec820ffff0f597039705457861f322f19'
ts.set_token(token)
pro = ts.pro_api()

stock_data = pd.read_csv('git_project/git_project/stock/kcb.csv').dropna()
stock_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
stock_code = np.array(stock_data['WINDCODE'].drop_duplicates())
names = locals()
for code in stock_code: names['df' + str(code)] = stock_data[stock_data['WINDCODE'] == code]

def cal_smb_hml(df):
    # 划分大小市值公司
    df['SB'] = df['circ_mv'].map(lambda x: 'B' if x >= df['circ_mv'].median() else 'S')
    # 求账面市值比：PB的倒数
    df['BM'] = 1 / df['pb']
    # 划分高、中、低账面市值比公司
    border_down, border_up = df['BM'].quantile([0.3, 0.7])
    border_down, border_up
    df['HML'] = df['BM'].map(lambda x: 'H' if x >= border_up else 'M')
    df['HML'] = df.apply(lambda row: 'L' if row['BM'] <= border_down else row['HML'], axis=1)
    # 组合划分为6组
    df_SL = df.query('(SB=="S") & (HML=="L")')
    df_SM = df.query('(SB=="S") & (HML=="M")')
    df_SH = df.query('(SB=="S") & (HML=="H")')
    df_BL = df.query('(SB=="B") & (HML=="L")')
    df_BM = df.query('(SB=="B") & (HML=="M")')
    df_BH = df.query('(SB=="B") & (HML=="H")')
    # 计算各组收益率
    R_SL = (df_SL['pct_chg'] * df_SL['circ_mv'] / 100).sum() / df_SL['circ_mv'].sum()
    R_SM = (df_SM['pct_chg'] * df_SM['circ_mv'] / 100).sum() / df_SM['circ_mv'].sum()
    R_SH = (df_SH['pct_chg'] * df_SH['circ_mv'] / 100).sum() / df_SH['circ_mv'].sum()
    R_BL = (df_BL['pct_chg'] * df_BL['circ_mv'] / 100).sum() / df_BL['circ_mv'].sum()
    R_BM = (df_BM['pct_chg'] * df_BM['circ_mv'] / 100).sum() / df_BM['circ_mv'].sum()
    R_BH = (df_BH['pct_chg'] * df_BH['circ_mv'] / 100).sum() / df_BH['circ_mv'].sum()
    # 计算SMB, HML并返回
    smb = (R_SL + R_SM + R_SH - R_BL - R_BM - R_BH) / 3
    hml = (R_SH + R_BH - R_SL - R_BL) / 2
    return smb, hml


data = []
df_cal = pro.trade_cal(start_date='20170101', end_date='20190110')
df_cal = df_cal.query('(exchange=="SSE") & (is_open==1)')
for date in df_cal.cal_date:
    df_daily = pro.daily(trade_date=date)
    df_basic = pro.daily_basic(trade_date=date)
    df = pd.merge(df_daily, df_basic, on='ts_code', how='inner')
    smb, hml = cal_smb_hml(df)
    data.append([date, smb, hml])
    print(date, smb, hml)

df_tfm = pd.DataFrame(data, columns=['trade_date', 'SMB', 'HML'])
df_tfm['trade_date'] = pd.to_datetime(df_tfm.trade_date)
df_tfm = df_tfm.set_index('trade_date')
df_tfm.to_csv('df_three_factor_model.csv')
df_tfm.head()