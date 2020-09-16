import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
import time
###########
# 找双金叉的时间
# 流程：设置参数、寻找特点、画图验证
# 可修改参数：start_date、end_date：开始，结束时间点； random_stock：随意验证的股票代码
# 结果：final_code：所有符合特点的股票代码
###########
# 设置参数
# start_date、end_date：开始，结束时间点，（格式：年月日6位数字）
start_date = '20180401'
end_date = '20181025'
code_file_name = 'git_project/git_project/stock/stock_code.csv'

ts.set_token('fe2fd196465dab93ba1c5f95d7908d09aab54e7d274a17cb05b7a580') # 数据接口token
###########
# 读取科创板所有代码，并查找符合双金叉特点的股票
# final_code：最后符合结果的股票
stock_code = pd.read_csv(code_file_name)
time_start = time.time()
final_code = []
final_count = []
for i in range(stock_code.shape[0]):
    pro = ts.pro_api()
    df = ts.pro_bar(ts_code=stock_code['code'][i], start_date=start_date, end_date=end_date, ma=[13, 28]).dropna()
    df['dif'] = df['ma13'] - df['ma28']
    df['std'] = df['dif'].map(lambda x: 1 if x >= 0 else -1)
    df['gc'] = df['std'] * df['std'].shift(1)
    count = df.loc[(df['gc'] < 0) & (df['dif'] < 0), 'gc'].count()
    if count == 2:
        final_code.append(stock_code['code'][i])
        final_count.append(count)
    else:
        continue

time_end = time.time()
print('time cost', time_end-time_start)
print(len(final_code))


final_code = pd.DataFrame(final_code)
final_code.to_csv('git_project/git_project/stock/final_code.csv')
# data = pd.DataFrame(columns=['code','count'], data=[[final_code],[final_count]])

#################
# 画图画图： 13日均线及28日均线
random_stock = '300107.SZ' # 从final code里随意选择一只股票代码

pro = ts.pro_api()
yz = ts.pro_bar(ts_code=random_stock, start_date=start_date, end_date=end_date, ma=[13, 28]).dropna()

plt.figure(figsize=(50,10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(yz['ma13'].to_list()[::-1],color="b",label="13")
plt.plot(yz['ma28'].to_list()[::-1],color="r",label="28")
plt.legend()
plt.xlabel("Time")
plt.ylabel("MA")
plt.title(random_stock)
plt.show()

################

