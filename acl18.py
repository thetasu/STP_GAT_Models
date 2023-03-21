# Author: LC
# Date Time: 2020/12/11 10:12
# File Description:

from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import logging
import os

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


def get_base_info(useRaw=True):
    base_info = DataFrame()
    if useRaw:
        # 获取 ./data/ACL18/ 文件夹下有哪些文件
        file_names = os.listdir('./data/ACL18/')
    else:
        # 获取 ./data/ACL18-P/ 文件夹下有哪些文件
        file_names = os.listdir('./data/ACL18-P')
    # 对文件进行排序，要优雅
    file_names.sort()
    # 循环读取每个文件
    count = 0
    for file_name in file_names:
        count += 1
        if useRaw:
            info = pd.read_csv('./data/ACL18/' + file_name)
        else:
            names = ['Date', 'Movement', 'Open', 'High', 'Low', 'Close', 'Volume']
            info = pd.read_csv('./data/ACL18-P/' + file_name, sep='\t', names=names)
            info = info.iloc[::-1].reset_index(drop=True)
        # 判断是否按日期升序排列，以防出错
        assert info.iloc[0, 0] < info.iloc[1, 0]
        # 原始数据中包含 null 我也是没想到
        info.dropna(axis=0, how='any', inplace=True)
        # 剔除 volume 为 0 的数据
        info = info.loc[info['Volume'] != 0, :]
        code = file_name.split('.')[0]
        info.insert(1, 'Code', code)
        base_info = pd.concat([base_info, info])
        logger.debug('(%02d/%02d) Data shape: (%04d, %02d) Current stock: %s' % (
            count, len(file_names), info.shape[0], info.shape[1], code))
    logger.info('Get base info done.')
    return base_info


def get_dataset(base_info):
    count, codes = 0, len(set(base_info['Code']))
    #
    for code, info in base_info.groupby(by='Code'):
        count += 1
        # 生成 c_open、c_high、c_low、n_close、n_adj_close 特征
        low = info['Low'].astype(np.float).values.tolist()
        open = info['Open'].astype(np.float).values.tolist()
        high = info['High'].astype(np.float).values.tolist()
        close = info['Close'].astype(np.float).values.tolist()
        adj_close = info['Adj Close'].astype(np.float).values.tolist()
        c_open, c_high, c_low, n_close, n_adj_close = [None], [None], [None], [None], [None]
        for i in range(1, len(close)):
            c_low.append('%.6f' % (100 * (low[i] / close[i] - 1)))
            c_open.append('%.6f' % (100 * (open[i] / close[i] - 1)))
            c_high.append('%.6f' % (100 * (high[i] / close[i] - 1)))
            n_close.append('%.6f' % (100 * (close[i] / close[i - 1] - 1)))
            n_adj_close.append('%.6f' % (100 * (adj_close[i] / adj_close[i - 1] - 1)))
        base_info.loc[base_info['Code'] == code, 'c_low'] = c_low
        base_info.loc[base_info['Code'] == code, 'c_open'] = c_open
        base_info.loc[base_info['Code'] == code, 'c_high'] = c_high
        base_info.loc[base_info['Code'] == code, 'n_close'] = n_close
        base_info.loc[base_info['Code'] == code, 'n_adj_close'] = n_adj_close
        # 生成 5_day、10_day、15_day、20_day、25_day、30_day 特征
        day = [[None] * 29, [None] * 29, [None] * 29, [None] * 29, [None] * 29, [None] * 29]
        for i in range(29, len(adj_close)):
            for j, T in enumerate([5, 10, 15, 20, 25, 30]):
                day[j].append('%.6f' % (100 * (sum(adj_close[i - T + 1: i + 1]) / T / adj_close[i] - 1)))
        base_info.loc[base_info['Code'] == code, '5_day'] = day[0]
        base_info.loc[base_info['Code'] == code, '10_day'] = day[1]
        base_info.loc[base_info['Code'] == code, '15_day'] = day[2]
        base_info.loc[base_info['Code'] == code, '20_day'] = day[3]
        base_info.loc[base_info['Code'] == code, '25_day'] = day[4]
        base_info.loc[base_info['Code'] == code, '30_day'] = day[5]
        # 生成标签
        label = []
        for i in range(1, len(n_adj_close)):
            if n_adj_close[i] is None:
                label.append(None)
            elif float(n_adj_close[i]) <= -0.005 * 100:
                label.append('-1')
            elif float(n_adj_close[i]) >= 0.0055 * 100:
            # elif float(n_adj_close[i]) > 0.0055 * 100:
                label.append('+1')
            else:
                label.append('0')
        label.append(None)
        base_info.loc[base_info['Code'] == code, 'label'] = label
        logger.debug('(%02d/%02d) Get dataset...' % (count, codes))
    dataset = base_info.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    logger.info('Get dataset done.')
    return dataset


if __name__ == '__main__':
    # 获取基础数据
    base_info = get_base_info(useRaw=True)  # TODO
    # 基于基础数据创建特征
    dataset = get_dataset(base_info)
    # 剔除含有 np.nan 的数据
    dataset.dropna(axis=0, how='any', inplace=True)
    # 保存数据
    dataset.to_csv('./data/ACL-V2.csv', header=True, index=False)  # TODO
