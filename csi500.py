# Author: LC
# Date Time: 2020/12/9 15:29
# File Description:

from pandas.core.frame import DataFrame
import baostock as bs
import tushare as ts
import pandas as pd
import numpy as np
import logging
import os

pro = ts.pro_api('5e440fc23c7094ffebec94e06607adaf3a47cb337c6aeb63ba5fad71')

filepath = os.path.join(os.path.split(__file__)[0], 'logs', os.path.split(__file__)[-1].split('.')[0] + '.log')

# Step 1: 创建一个 logger。
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Step 2: 创建一个 handler，用于将日志输出至控制台。
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Step 3: 创建一个 handler，用于将日志写入文件。
file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
# Step 4: 定义 handler 的输出格式。
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)03d] - %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Step 5: 将 logger 添加到 handler 中。
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_con_codes():
    con_codes, df = set(), DataFrame()
    # 获取 20190701 至 20200630 期间，中证500指数每个月的成分股代码
    df = pd.concat([df, pro.index_weight(index_code='000905.SH', start_date='20190701', end_date='20191231').iloc[
                        ::-1].reset_index(drop=True)])
    df = pd.concat([df, pro.index_weight(index_code='000905.SH', start_date='20200101', end_date='20200630').iloc[
                        ::-1].reset_index(drop=True)])
    # 保留 20190701 至 20200630 期间未发生过变动的成分股代码
    for trade_date, group in df.groupby(by='trade_date'):
        if len(con_codes) == 0:
            con_codes = set(group.con_code)
        else:
            con_codes = con_codes & set(group.con_code)
    logger.info('Get con codes done.')
    return con_codes


def get_con_price_info(con_codes):
    bs.login()
    con_price_info, count = DataFrame(), 0
    for con_code in con_codes:
        count += 1
        con_code = con_code[7:9] + '.' + con_code[0:6]
        rs = bs.query_history_k_data_plus(con_code, "date, time, code, open, high, low, close, volume, amount",
                                          start_date='2019-07-01', end_date='2020-06-30', frequency='5')
        info = []
        while rs.error_code == '0' and rs.next():
            info.append(rs.get_row_data())
        info = pd.DataFrame(info, columns=rs.fields)
        con_price_info = pd.concat([con_price_info, info])
        logger.debug('(%03d/%03d) Get %s price info done.' % (count, len(con_codes), con_code))
    # 截取 code 前的所属证券交易所
    con_price_info['code'] = con_price_info['code'].str[3:]
    # 截取 time 尾部冗余的零
    con_price_info['time'] = con_price_info['time'].str[:12]
    logger.info('Get con price info done.')
    bs.logout()
    return con_price_info


def delete_invalid_info(con_price_info):
    count, valid_con_price_info = 0, DataFrame()
    for code, info in con_price_info.groupby(by='code'):
        if info.shape[0] == 11664:
            valid_con_price_info = pd.concat([valid_con_price_info, info])
        else:
            count += 1
    logger.info('Delete invalid info done.')
    print('Delete %d stocks with missing data.' % count)
    return valid_con_price_info


def add_useful_attributes(con_price_info, add_vol_ratio=False, T=10):
    # 为股票添加 change 属性
    con_price_info['change'] = np.nan
    for code, info in con_price_info.groupby(by='code'):
        close, change = info['close'].astype(np.float).values.tolist(), []
        change.append(None)
        for i in range(1, len(close)):
            change.append('%.2f' % (close[i] - close[i - 1]))
        con_price_info.loc[con_price_info['code'] == code, 'change'] = change
    logger.info('Attribute change added succeed.')
    # 为股票添加 pct_chg 属性
    con_price_info['pct_chg'] = np.nan
    for code, info in con_price_info.groupby(by='code'):
        close, change, pct_chg = info['close'].astype(np.float).values.tolist(), info['change'].astype(
            np.float).values.tolist(), []
        pct_chg.append(None)
        for i in range(1, len(close)):
            pct_chg.append('%.6f' % float(change[i] / close[i - 1]))
        con_price_info.loc[con_price_info['code'] == code, 'pct_chg'] = pct_chg
    logger.info('Attribute pct_chg added succeed.')
    # 为股票添加 avg_vol 属性
    con_price_info['avg_vol'] = np.nan
    for code, info in con_price_info.groupby(by='code'):
        volume, avg_vol = info['volume'].astype(np.float).values.tolist(), []
        for _ in range(T - 1):
            avg_vol.append(None)
        for i in range(T - 1, len(volume)):
            avg_vol.append('%.2f' % float(np.mean(volume[i - T + 1: i + 1])))
        con_price_info.loc[con_price_info['code'] == code, 'avg_vol'] = avg_vol
    logger.info('Attribute avg_vol added succeed.')
    if add_vol_ratio:
        # 为股票添加 vol_ratio 属性  # TODO: 是否可以考虑换成 avg_vol_ratio
        con_price_info['vol_ratio'] = np.nan
        for time, info in con_price_info.groupby(by='time'):
            volume = info['volume'].astype(np.float).values
            total_volume = np.sum(volume)
            vol_ratio = np.around(volume / total_volume, decimals=6)
            con_price_info.loc[con_price_info['time'] == time, 'vol_ratio'] = vol_ratio
        logger.info('Attribute vol_ratio added succeed.')
    return con_price_info


def transform_into_temporal_info(con_price_info, num, add_vol_ratio=False, T=5):
    #
    if add_vol_ratio:
        attrs = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'pct_chg', 'vol_ratio']
    else:
        attrs = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'pct_chg']
    # 提前设置好布局
    for i in range(1, T):
        for attr in attrs:
            con_price_info['pre_' + str(i) + '_' + attr] = np.nan
    #
    count = 0
    for code, info in con_price_info.groupby(by='code'):
        count += 1
        for attr in attrs:
            columns = ['pre_' + str(i) + '_' + attr for i in range(1, T)]
            base_info, temporal_info = info[attr].values.tolist(), []
            for _ in range(T - 1):
                temporal_info.append([None] * (T - 1))
            for i in range(T - 1, len(info)):
                temporal_info.append([base_info[i - j] for j in range(1, T)])
            con_price_info.loc[con_price_info['code'] == code, columns] = temporal_info
        logger.debug('(%03d/%03d) Transform %s into temporal info done.' % (count, num, code))
    logger.info('Transform into temporal info done.')
    return con_price_info


def get_con_industry_info(con_codes):
    # 按理说应该根据股票代码去查找其对应的申万行业，但是在调用 index_member() 方法获取时，发现该方法只能返回股票所对应申万行业的代码，因而采取下
    # 述方法：先调用 index_classify() 方法获取所有的申万行业以及各自对应的代码，然后调用 index_member() 方法获取每个申万行业包含了哪些股票，
    # 最后根据上述信息获取股票对应的申万行业。
    # con_industry_info, count = defaultdict(list), 0  # count 用于对申万行业涉及的股票进行计数
    # for index_code, industry_name in pro.index_classify(level='L1', src='SW')[['index_code', 'industry_name']].values.tolist():
    #     for con_code in pro.index_member(index_code=index_code)['con_code'].values.tolist():
    #         count += 1
    #         if con_code in con_codes:
    #             con_industry_info[con_code].append(industry_name)  # 经过验证，每只股票只对应了一个行业，因此对代码进行了简化。
    # # print(count)  # 3735
    con_industry_info = {}
    for index_code, industry_name in pro.index_classify(level='L1', src='SW')[
        ['index_code', 'industry_name']].values.tolist():
        for con_code in pro.index_member(index_code=index_code)['con_code'].values.tolist():
            if con_code in con_codes:
                con_industry_info[con_code[0:6]] = industry_name
    logger.info('Get con industry info succeed.')
    return con_industry_info


def get_con_info(con_price_info, con_industry_info):
    con_price_info['industry'] = np.nan
    for code, _ in con_price_info.groupby(by='code'):
        con_price_info.loc[con_price_info['code'] == code, 'industry'] = con_industry_info[code]
    logger.info('Get con info succeed.')
    return con_price_info


if __name__ == '__main__':

    # 获取 20190701 至 20200630 期间未发生过变动的成分股代码(共 405 个)
    con_codes = get_con_codes()

    # 获取 20190701 至 20200630 期间所有成分股的价格相关信息(共 4717728 条，与从 tushare 获取的数据相比，少了 pre_close、change 和 pct_change)
    con_price_info = get_con_price_info(list(con_codes))

    # 删除缺少数据的成分股，理论上单只股票应该有 11664 条数据
    con_price_info = delete_invalid_info(con_price_info)

    # 为股票添加 change, pct_chg, avg_vol 属性
    con_price_info = add_useful_attributes(con_price_info)

    # 基于滑动窗口将每一条数据处理为时序性数据
    con_price_info = transform_into_temporal_info(con_price_info, len(set(con_price_info['code'])))

    # 获取 20190701 至 20200630 期间所有成分股的行业相关信息
    con_industry_info = get_con_industry_info(con_codes)

    # 将成分股的价格相关信息和行业相关信息融合在一起
    con_info = get_con_info(con_price_info, con_industry_info)

    # 为每只股票增加 label 属性
    T = 5  # 以未来 5 天的收盘价的平均值作为对比
    con_info['label'] = np.nan
    for code, info in con_info.groupby(by='code'):
        close, label = info['close'].astype(np.float).values.tolist(), []
        for i in range(0, len(close) - T):
            next_close = float('%.2f' % np.mean(close[i + 1: i + T + 1]))  # TODO: next_close 是否需要保存两位小数，若需要，该调用哪种方法
            if close[i] < next_close:
                label.append('up')
            elif close[i] > next_close:
                label.append('down')
            else:
                label.append('equal')
        for i in range(0, T):
            label.append(None)
        con_info.loc[con_info['code'] == code, 'label'] = label
    logger.info('Attribute label added succeed.')

    # 剔除含有 np.nan 的数据
    con_info.dropna(axis=0, how='any', inplace=True)

    con_info.to_csv('./data/CSI500-20190701-20200630-5M-Latest.csv', header=True, index=False)
