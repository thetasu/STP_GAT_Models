# Author: SHY
# Date Time: 2020/12/11 10:12
# File Description: splicing time series data for ACL-V2.csv
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import numpy as np
import logging
import os

filepath = os.path.join(os.path.split(__file__)[0], 'logs', datetime.datetime.today().strftime('%Y %m %d') + ' ' +
                        os.path.split(__file__)[-1].split('.')[0] + '.log')

# # Step 1: 创建一个 logger。
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# # Step 2: 创建一个 handler，用于将日志输出至控制台。
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# # Step 3: 创建一个 handler，用于将日志写入文件。
# file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
# file_handler.setLevel(logging.INFO)
# # Step 4: 定义 handler 的输出格式。
# formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)03d] - %(levelname)s: %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
# # Step 5: 将 logger 添加到 handler 中。
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# seq_len 为拼接长度
file_path = './data/ACL-V2.csv'
data = pd.read_csv(file_path)

# 遍历股票代码，按照时间顺序对每支股票数据进行拼接
code_list = data['Code'].values.tolist()
code_list = list(set(code_list))
code_list.sort() # 按照字母排序股票代码
result_list = [] # 存放所有待整合数据，最后纵向拼接成Dataframe
for Code in code_list:
    stock_data = data[data["Code"] == Code]
    seq_len = 5
    for j in range(len(stock_data) - seq_len + 1):
        unslice_data_list = [] # 存放每个时间点特征
        for i in range(seq_len):
            if i == seq_len-1:
                unslice_data = DataFrame(stock_data.values[i+j:i+j+1,0:14]) # 保留拼接最后一条数据的Date,Code,Label
                unslice_data_list.append(unslice_data)
            else:
                unslice_data = DataFrame(stock_data.values[i+j:i+j+1,2:13])
                unslice_data_list.append(unslice_data)

        # 特征横向拼接， 只保留最后一条待拼接特征的Date、Code、Label 作为序列的信息
        result = pd.concat(unslice_data_list,axis=1)
        ori_col = ['c_low', 'c_open', 'c_high', 'n_close', 'n_adj_close', '5_day', '10_day', '15_day', '20_day', '25_day',
                   '30_day']
        ori_col_total = ['Date', "Code", 'c_low', 'c_open', 'c_high', 'n_close', 'n_adj_close', '5_day', '10_day', '15_day',
                         '20_day', '25_day', '30_day']

        # 重命名Dataframe每一列
        append_col = []
        for i in range(1, seq_len):
            # 判断是不是最后一条待拼接数据，如果是，就保留Date和CodeL两个字段
            if i == seq_len - 1:
                append_col.append('Date')
                append_col.append('Code')
            for col in ori_col:
                new_col_name = "Pre_" + str(i) + "_" + col
                append_col.append(new_col_name)
        final_col = ori_col + append_col
        final_col.append('label')
        result.columns = final_col
        df_id = result.Date
        result = result.drop('Date',axis=1)
        result.insert(0,'Date',df_id)
        df_id = result.Code
        result = result.drop('Code', axis=1)
        result.insert(1, 'Code', df_id)
        print(result)
        result_list.append(result)

clear_result = pd.concat(result_list,axis=0,ignore_index=True)
print(clear_result)

clear_result.to_csv('./data/ACL-V2-P.csv')
print('write success')

