import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

###########
# Para
start_date = '2020-01-03'
end_date = '2020-07-02'
lu_jing = 'git_project/git_project/stock/kcb.csv'

###########
# Buy
stock_data = pd.read_csv(lu_jing).dropna()
stock_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
stock_code = np.array(stock_data['WINDCODE'].drop_duplicates())
names = locals()
for code in stock_code: names['df' + str(code)] = stock_data[stock_data['WINDCODE'] == code]


def xy_data(eg_code):
    ori_df = names['df' + str(eg_code)]
    df = ori_df.copy()
    df.date = pd.to_datetime(ori_df.date)
    df = df.set_index('date')
    names['x_data' + str(eg_code)] = df[start_date:end_date]
    names['y_data' + str(eg_code)] = df[end_date:]
    return names['x_data' + str(eg_code)], names['y_data' + str(eg_code)]


def testcode():
    test_code = []
    for eg_code in stock_code:
        df = names['df' + str(eg_code)]
        date = df['date'].tolist()
        if start_date in date:
            test_code.append(eg_code)
    return test_code


def selectedcode():
    selected_code = []
    for eg_code in testcode():
        x_data = xy_data(eg_code)[0]
        x_data['dif'] = x_data['MA_13'] - x_data['MA_28']
        x_data['sign'] = pd.Series(x_data.dif.map(lambda x: 1 if x >= 0 else -1))
        x_data['cross_amount'] = pd.Series(x_data.sign * x_data.sign.shift(1))
        choose1 = x_data.cross_amount == -1
        choose2 = x_data.dif > 0
        point = x_data[choose1 & choose2]
        count = point.shape[0]
        if count >= 2:
            if x_data.index[-1] == point.index[-1]:
                rate = (point.CLOSE.to_list()[-1] - point['CLOSE'].to_list()[-2])/point['CLOSE'].to_list()[-2]
                if rate < 0.1:
                    selected_code.append(eg_code)
                else:
                    continue
        else:
            continue
    return selected_code


#########
# Star Rank
def star_rank():
    # 设置打分规则：现金流 > 5，3分；资产负债率 < 60，减1分；资产负债率 < 30，加1分
    # 读取数据（已通过R整理后，显示资产收益率ROE，以及资产回报率ROA分数）
    # 规则： ROE > 15，每大1加1分，v.v.； ROA > 14，每大1加1分，v.v.
    stock_star = pd.read_csv('git_project/git_project/stock/stock_star.csv')
    stock_star['score'] = stock_star.ROE_star + stock_star.ROA_star
    other_score = np.zeros(stock_star.shape[0])
    for i in range(stock_star.shape[0]):
        if stock_star['cashflow'][i] > 5:
            other_score[i] += 3
        if stock_star['liabilities'][i] <30:
            other_score[i] += 1
        if stock_star['liabilities'][i] >60:
            other_score[i] -= 1
        else:
            continue
    stock_star['score'] += other_score

    # 剔除
    # 规则： 剔除资产负债率 > 70，现金流 < 0
    drop1 = stock_star['liabilities'] > 70
    drop2 = stock_star['cashflow'] < 0
    stock_star.drop(stock_star[drop1 | drop2].index, inplace=True)

    # Star
    # 规则： 前100名5星，200名4星，300名3星，400名2星，其余1星
    sorted_df = stock_star.sort_values(by='score', axis=0, ascending=False)
    star = np.zeros(stock_star.shape[0])
    star[:100] = 5
    star[100:200] = 4
    star[200:300] = 3
    star[300:400] = 2
    star[400:] = 1
    sorted_df['star'] = star
    star_rank = pd.DataFrame(sorted_df['code'])
    star_rank['star'] = star
    # star_rank.to_csv('git_project/git_project/stock/star_rank.csv')

    # star_rank with selected
    selected_star = star_rank.loc[star_rank.code.isin(selectedcode())]
    return star_rank, selected_star


#######
# Calculate Earning
def earn():
    Earning = []
    for i in range(len(selectedcode())):
        y_data = xy_data(selectedcode()[i])[1]
        earning = (y_data['CLOSE'].tolist()[-1] - y_data['CLOSE'].tolist()[0]) / y_data['CLOSE'].tolist()[0]
        Earning.append(earning)
        if i == 0:
            y_close = pd.DataFrame(y_data['CLOSE'])
        else:
            y_close = pd.concat([y_close,y_data['CLOSE']], axis=1)
    return Earning, y_close


######
# Plot
def plot_one(random_stock):
    plt.figure(figsize=(50,10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(xy_data(random_stock)[0]['MA_13'],color="b",label="MA13")
    plt.plot(xy_data(random_stock)[0]['MA_28'],color="r",label="MA28")
    plt.legend(fontsize=50)
    plt.title(random_stock,fontsize=50)
    plt.show()
    return


def plot_all(num_1, num_2):
    plt.figure(figsize=(50,35))
    for i in range(len(selectedcode())):
        plt.subplot(num_1,num_2, (i + 1))
        plt.plot(xy_data(selectedcode()[i])[0]['MA_13'], color="b", label="MA13")
        plt.plot(xy_data(selectedcode()[i])[0]['MA_28'], color="r", label="MA28")
        plt.legend(fontsize=50)
        plt.title(selectedcode()[i],fontsize=50)
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


##################
# Call Functions
selectedcode() # all code
len(selectedcode()) # number of all code
star_rank()[1] # all code with rank
# Plot
plot_one(selectedcode()[0]) # 画第一支股票的图
plot_all(1, 1) #画所有股票的图（可以根据len改括号里面的数）

