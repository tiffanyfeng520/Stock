import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd

##########
# 打分
# 流程：设置参数，设置打分规则，剔除，评星，合并上一个双金叉股票，验证画图
# 可修改参数：start_date、end_date：开始，结束时间点； random_stock：随意验证的股票代码
# 结果：final_star：股票评星结果；final：与双金叉合并后评星结果
##########
# 设置参数
# start_date、end_date：开始，结束时间点，（格式：年月日6位数字）
start_date = '20190101'
end_date = '201901225'
star_file_name = 'git_project/git_project/stock/stock_star.csv'

####
# 设置打分规则：现金流 > 5，3分；资产负债率 < 60，减1分；资产负债率 < 30，加1分
# 读取数据（已通过R整理后，显示资产收益率ROE，以及资产回报率ROA分数）
# 规则： ROE > 15，每大1加1分，v.v.； ROA > 14，每大1加1分，v.v.

stock_star = pd.read_csv(star_file_name)
stock_star['score'] = stock_star.iloc[:,3] + stock_star.iloc[:,4]
other_score = np.zeros(stock_star.shape[0])
for i in range(stock_star.shape[0]):
    if stock_star['cashflow'][i] > 5:
        other_score[i] += 3
    if stock_star['liabilities'][i] <30:
        other_score[i] += 1
    elif stock_star['liabilities'][i] >60:
        other_score[i] -= 1
    else:
        continue

stock_star['score'] = stock_star['score'] + other_score

##########
# 剔除
# 规则： 剔除资产负债率 > 70，现金流 < 0
drop1 = np.where(stock_star['liabilities'] >70)
stock_star.drop(stock_star.index[drop1], inplace=True)
drop2 = np.where(stock_star['cashflow'] < 0)
stock_star.drop(stock_star.index[drop2], inplace=True)
##########
# 评星
# 规则： 前100名5星，200名4星，300名3星，200名2星，其余1星
sorted_df = stock_star.sort_values(by='score', axis=0, ascending=False)

star = np.zeros(stock_star.shape[0])

for i in range(stock_star.shape[0]):
    if i < 100:
        star[i] = 5
    elif i < 200:
        star[i] = 4
    elif i < 300:
        star[i] = 3
    elif i < 400:
        star[i] = 2
    else:
        star[i] = 1

sorted_df['star'] = star
star_rank = pd.DataFrame(sorted_df['code'])
star_rank['star'] = star
star_rank.to_csv('git_project/git_project/stock/star_rank.csv')



# sorted_df.to_excel("final_star.xlsx") # 最后结果可写进Excel文档，名字为"final_star"

############
# 双金叉与打分合并
final_code = pd.read_csv('git_project/git_project/stock/final_code.csv')
final_code = np.array(final_code)[:,-1].tolist()


dictionary = pd.DataFrame(sorted_df['code'])
dictionary['star'] = sorted_df['star']
dictionary_code = np.array(dictionary['code']).tolist()

final = dictionary.loc[dictionary['code'].isin(final_code)]
final = pd.DataFrame(final)
final.to_excel('final.xlsx') # 最后结果可写进Excel文档，名字为"final"


yz_zf = []
for i in range(len(final['code'])):
    pro = ts.pro_api()
    df = pro.daily(ts_code=final['code'][i], start_date='201901226', end_date='20200110')
    zf = (df['close'].max() - df['close'][0])/df['close'][0]
    yz_zf.append(zf)

##################
# 验证前35名画图

final = pd.read_excel("final.xlsx")
five = final.loc[final['star'] == 5]
five_code = np.array(five['code'])
ma13 = []
ma13 = pd.DataFrame(ma13)
ma28 = []
ma28 = pd.DataFrame(ma28)

for i in range(len(five_code)):
    pro = ts.pro_api()
    yz = ts.pro_bar(ts_code=five_code[i], start_date=start_date, end_date=end_date, ma=[13, 28]).dropna()
    ma13_cur = np.zeros(111)
    ma28_cur = np.zeros(111)
    ma13_cur[-len(yz['ma13'].to_list()[::-1]):] = yz['ma13'].to_list()[::-1]
    ma28_cur[-len(yz['ma28'].to_list()[::-1]):] = yz['ma28'].to_list()[::-1]
    ma13[str(five_code[i])] = ma13_cur
    ma28[str(five_code[i])] = ma28_cur

plt.figure(figsize=(50,35))
for i in range(23):
    plt.subplot(5, 5, (i + 1))
    plt.plot(ma13[str(five_code[i])], color="b", label="MA13")
    plt.plot(ma28[str(five_code[i])], color="r", label="MA28")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("MA")
    plt.title(five_code[i])
    plt.grid(True)

plt.tight_layout()
plt.show()


# len(five_code)
# df = ts.pro_bar(ts_code='600519.SH', start_date='20180101', end_date='20181011', factors='vr')


# pro = ts.pro_api()
# data = pro.stock_basic(exchange='', list_status='L',
#                        fields='ts_code,symbol,name,area,industry,market,list_date')
