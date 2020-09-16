import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
import mpl_finance as mpf
import pylab


##############
# Para
start_date = '2019-05-30'
end_date = '2020-07-14'
stock_b_code = '688028.SH'
MA1 = 13
MA2 = 28


###############
# Data
def dataset(eg_code, start_date, end_date):
    stock_data = pd.read_csv('git_project/git_project/stock/kcb.csv')
    stock_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    stock_code = np.array(stock_data['WINDCODE'].drop_duplicates())
    names = locals()
    for code in stock_code:
        names['df' + str(code)] = stock_data[stock_data['WINDCODE'] == code]
    ori_data = names['df' + str(eg_code)].dropna()
    data = ori_data.copy()
    data.loc[:,['date']] = pd.to_datetime(ori_data.date)
    data = data.set_index('date')
    return data[start_date:end_date]


def data_reshape():
    data = dataset(stock_b_code, start_date, end_date)
    # Data reshape
    data_reshape = data.reset_index()
    data_reshape['DateTime'] = mdates.date2num(data_reshape['date'].astype('M8[s]'))
    data_reshape = data_reshape[['DateTime', 'OPEN', 'HIGH', 'LOW', 'CLOSE']]
    return data_reshape


########
def label():
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                       fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='w')
    return


def volume(ax1, data, re_data):
    volumeMin = 0
    ax1v = ax1.twinx()
    ax1v.fill_between(re_data.DateTime.values, volumeMin, data.VOLUME.values, facecolor='#00ffe8', alpha=.4)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3 * data.VOLUME.values.max())
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')
    return


def get_macd(num_periods_fast=10, num_periods_slow=40, num_periods_macd=20):
    data = dataset(stock_b_code, start_date, end_date)
    # num_periods_fast = 10  # 快速EMA的时间周期，10
    # num_periods_slow = 40  # 慢速EMA的时间周期，40
    # num_periods_macd = 20  # MACD EMA的时间周期，20
    # K:平滑常数，常取2/(n+1)
    K_fast = 2 / (num_periods_fast + 1)  # 快速EMA平滑常数
    K_slow = 2 / (num_periods_slow + 1)  # 慢速EMA平滑常数
    K_macd = 2 / (num_periods_macd + 1)  # MACD EMA平滑常数

    ema_fast, ema_slow, ema_macd = 0, 0, 0
    ema_fast_values, ema_slow_values = [], []
    macd_values, macd_signal_values, MACD_hist_values = [], [], [],
    for close_price in data.CLOSE:
        if ema_fast == 0:  # 第一个值
            ema_fast = close_price
            ema_slow = close_price
        else:
            ema_fast = (close_price - ema_fast) * K_fast + ema_fast
            ema_slow = (close_price - ema_slow) * K_slow + ema_slow

        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)
        # MACD is fast_MA - slow_EMA
        macd = ema_fast - ema_slow
        if ema_macd == 0:
            ema_macd = macd
        else:
        # signal is EMA of MACD values
            ema_macd = (macd - ema_macd) * K_slow + ema_macd
        macd_values.append(macd)
        macd_signal_values.append(ema_macd)
    return ema_slow_values, ema_fast_values, macd_values, macd_signal_values


def add_macd(ax1, re_data):
    ax2 = plt.subplot2grid((6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
    fillcolor = '#00ffe8'
    # ema_slow_values = get_macd()[0]
    # ema_fast_values = get_macd()[1]
    macd_values = np.array(get_macd()[2])
    macd_signal_values = np.array(get_macd()[3])

    ax2.plot(re_data.DateTime.values, macd_values, color='#4ee6fd', lw=2)
    ax2.plot(re_data.DateTime.values, macd_signal_values, color='#e1edf9', lw=1)
    ax2.fill_between(re_data.DateTime.values, macd_values - macd_signal_values, 0, alpha=0.5, facecolor=fillcolor,
                     edgecolor=fillcolor)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    loc_list = ['bottom', 'top', 'left', 'right']
    for loc in loc_list:
        ax2.spines[loc].set_color("w")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('MACD', color='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)
    return


''''
def mark(ax1, data):
    # Mark
    data_reshape = data.reset_index()
    data_reshape['DateTime'] = mdates.date2num(data_reshape['date'].astype('M8[s]'))
    data = data_reshape
    data['dif'] = data.MA_13 - data.MA_28
    data['sign'] = pd.Series(data.dif.map(lambda x: 1 if x >= 0 else -1))
    data['cross_amount'] = pd.Series(data.sign * data.sign.shift(1))
    choose1 = data.cross_amount == -1
    choose2 = data.dif > 0
    point = data[choose1 & choose2]
    ax1.annotate('Gold', (data.DateTime.values[point.index], data.CLOSE[point.index]),
                 xytext=(0.8, 0.9), textcoords='axes fraction',
                 arrowprops=dict(facecolor='white', shrink=0.05),
                 fontsize=10, color='w',
                 horizontalalignment='right', verticalalignment='bottom')
    return
'''


def rsiFunc(array_list, periods=14):
    length = len(array_list)
    rsies = [np.nan] * length
    if length <= periods:
        return rsies
    up_avg = 0
    down_avg = 0

    first_t = array_list[:periods + 1]
    for i in range(1, len(first_t)):
        if first_t[i] >= first_t[i - 1]:
            up_avg += first_t[i] - first_t[i - 1]
        else:
            down_avg += first_t[i - 1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    rs = up_avg / down_avg
    rsies[periods] = 100 - 100 / (1 + rs)

    for j in range(periods + 1, length):
        up = 0
        down = 0
        if array_list[j] >= array_list[j - 1]:
            up = array_list[j] - array_list[j - 1]
            down = 0
        else:
            up = 0
            down = array_list[j - 1] - array_list[j]
        up_avg = (up_avg * (periods - 1) + up) / periods
        down_avg = (down_avg * (periods - 1) + down) / periods
        rs = up_avg / down_avg
        rsies[j] = 100 - 100 / (1 + rs)
    return np.array(rsies)


def RSI(ax1, re_data):
    # plot an RSI indicator on top
    ax0 = plt.subplot2grid((6, 4), (0, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
    rsi = rsiFunc(re_data.CLOSE.values)
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'
    ax0.plot(re_data.DateTime.values, rsi, rsiCol, linewidth=1.5)
    ax0.axhline(70, color=negCol)
    ax0.axhline(30, color=posCol)
    ax0.fill_between(re_data.DateTime.values, rsi, 70, where=(rsi >= 70), facecolor=negCol,
                     edgecolor=negCol, alpha=0.5)
    ax0.fill_between(re_data.DateTime.values, rsi, 30, where=(rsi <= 30), facecolor=posCol,
                     edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([30, 70])
    ax0.yaxis.label.set_color("w")
    loc_list = ['bottom', 'top', 'left', 'right']
    for loc in loc_list:
        ax0.spines[loc].set_color("#5998ff")
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')
    plt.setp(ax0.get_xticklabels(), visible=False)
    return


def plot():
    data = dataset(stock_b_code, start_date, end_date)
    re_data = data_reshape()
    # Plot
    plt.figure(facecolor='#07000d', figsize=(15, 10))
    ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, facecolor='#07000d')
    mpf.candlestick_ohlc(ax1, re_data.values, width=.6, colorup='#ff1717', colordown='#53c156')
    label1 = 'MA ' + str(MA1); label2 = 'MA ' + str(MA2)
    ax1.plot(re_data.DateTime.values, data.MA_13, '#e1edf9', label=label1, linewidth=1.5)
    ax1.plot(re_data.DateTime.values, data.MA_28, '#4ee6fd', label=label2, linewidth=1.5)
    ax1.grid(True, color='w')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("w")
    loc_list = ['bottom', 'top', 'left', 'right']
    for loc in loc_list:
        ax1.spines[loc].set_color("w")
    ax1.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price and Volume')

    # Call Functions
    label()
    volume(ax1, data, re_data)
    add_macd(ax1, re_data)
    RSI(ax1, re_data)
    # mark(ax1, data)

    # Show
    plt.suptitle(stock_b_code, color='w')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    plt.show()
    return

plot()


