from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import pandas as pd
import numpy as np
import itertools
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def transform_into_temporal_info(dataset, T=5):
    #
    attrs = ['c_low', 'c_open', 'c_high', 'n_close', 'n_adj_close', '5_day', '10_day', '15_day', '20_day', '25_day', '30_day']
    # 提前设置好布局
    for i in range(1, T):
        for attr in attrs:
            dataset['Pre_' + str(i) + '_' + attr] = np.nan
    #
    for attr in attrs:
        columns = ['Pre_' + str(i) + '_' + attr for i in range(1, T)]
        base_info, temporal_info = dataset[attr].astype(np.float).values.tolist(), []
        for _ in range(T - 1):
            temporal_info.append([None] * (T - 1))
        for i in range(T - 1, len(dataset)):
            temporal_info.append([base_info[i - j] for j in range(1, T)])
        dataset.loc[:, columns] = temporal_info
    # 移动一下标签的位置，显得美观
    label = dataset['label']
    dataset.drop(labels=['label'], axis=1, inplace=True)
    dataset.insert(57, 'label', label)
    return dataset


def load_data(useACL=True):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))
    #
    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize_features(features)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    #
    # adj = torch.FloatTensor(np.array(adj.todense()))
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    #
    # return adj, features, labels, idx_train, idx_val, idx_test

    results = {}  # 用于保存返回结果
    # 加载数据集
    if useACL:
        datasets = pd.read_csv('./data/ACL-V2.csv', dtype=np.str)  # TODO
        # datasets = pd.read_csv('./data/ACL-V2-P.csv',index_col=0)  # TODO
        # datasets = pd.read_csv('./data/ACL-V2-P-30day.csv',index_col=0)  # TODO

    else:
        datasets = pd.read_csv('./data/CSI500-SUB5-20190701-20200630-5M-Done.csv', dtype=np.str)
    # 按 code 进行分组
    if useACL:
        by = 'Code'
    else:
        by = 'code'
    count, num = 0, len(set(datasets[by]))
    for code, dataset in datasets.groupby(by=by):
        # #
        dataset = dataset.loc[('2014-01-01' <= dataset['Date']) & (dataset['Date'] <= '2016-01-01'), :]
        if code == 'GMRE':
            continue
        results[code] = []
        # 剔除含有 nan 的行
        dataset.dropna(axis=0, how='any', inplace=True)
        # 对特征进行标准化
        if useACL:
            dataset.iloc[:, 2:13] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:13])  # TODO
            # dataset.iloc[:, 2:57] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:57])  # TODO
            # dataset.iloc[:, 2:112] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:112])  # TODO
            # dataset.iloc[:, 2:332] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:332])  # TODO

        else:
            dataset.iloc[:, 3:44] = MinMaxScaler().fit_transform(dataset.iloc[:, 3:44])  # TODO
        # 剔除标签为 0 的数据
        # dataset = dataset.loc[(dataset['label'] != '0') & (dataset['Code'] != 'GMRE'), :]
        dataset = dataset.loc[dataset['label'] != '0', :]
        # dataset = dataset.loc[dataset['label'] != 0, :]

        # 分段截取数据
        step = 2000
        for index in range(0, dataset.shape[0], step):
            count += 1
            #
            data = dataset.iloc[index:index + step, :]
            # 对数据集进行切分
            date = data['Date'].values.tolist()
            get_train_size = False
            for i in range(len(date)):
                if not get_train_size and date[i] > '2015-08-01':
                    train_size = i
                    get_train_size = True
                elif get_train_size and date[i] > '2015-10-01':
                    val_size = i
                    break
            idx_train, idx_val, idx_test = range(train_size), range(train_size, val_size), range(val_size, len(date))
            # idx_train, idx_val, idx_test = range(train_size), range(train_size, val_size), range(val_size, val_size+1)
            # idx_train, idx_val, idx_test = range(data.shape[0] - 120), range(data.shape[0] - 120, data.shape[0] - 60), range(data.shape[0] - 60, data.shape[0])
            #
            idx_features_labels = data.values
            # 提取 idx
            if useACL:
                idx = np.array(idx_features_labels[:, 0], dtype=np.str)
            else:
                idx = np.array(idx_features_labels[:, 1], dtype=np.str)
            idx_map = {j: i for i, j in enumerate(idx)}
            # 提取 features
            if useACL:
                features = sp.csr_matrix(idx_features_labels[:, 2:13], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:57], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:112], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:332], dtype=np.float)  # TODO
            else:
                features = sp.csr_matrix(idx_features_labels[:, 3:44], dtype=np.float)
                # features = sp.csr_matrix(idx_features_labels[:, 2:57], dtype=np.float)  # TODO

            # 提取 labels
            labels = encode_onehot(idx_features_labels[:, -1])
            # 基于上述数据构建邻接矩阵
            edges_unordered = []
            for _, group in data.iloc[0: train_size, :].groupby(by='label'):
            # for _, group in data.groupby(by='label'):
                #
                if useACL:
                    # edges_unordered.extend(list(itertools.product(group['Date'].values, repeat=2)))
                    # edges_unordered.extend(list(itertools.combinations(group['Date'].values, 2)))
                    # edges_unordered.extend(list(itertools.combinations_with_replacement(group['Date'].values, 2)))
                    # 只考虑 100 天以内的
                    date = group['Date'].values
                    for i in range(0, len(date)):
                        edges_unordered.extend(list(itertools.combinations(date[i: i + 10], 2)))  # 不包含自己到自己的边
                    edges_unordered.extend(list(zip(date, date)))  # 自己到自己的边
                else:
                    edges_unordered.extend(list(itertools.product(group['time'].values, repeat=2)))  # TODO
            edges_unordered = np.array(edges_unordered)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
                edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
            # 对邻接矩阵进行标准化
            adj = normalize_adj(adj)  # 无需加上单位对角矩阵
            print('adj.shape:{}'.format(adj.shape))
            #
            features = torch.tensor(np.array(features.todense()), dtype=torch.float)  # TODO
            adj = torch.tensor(np.array(adj.todense()), dtype=torch.float)  # TODO
            labels = torch.tensor(np.where(labels)[1], dtype=torch.long)  # TODO
            #
            idx_train = torch.tensor(idx_train, dtype=torch.long)  # TODO
            idx_val = torch.tensor(idx_val, dtype=torch.long)  # TODO
            idx_test = torch.tensor(idx_test, dtype=torch.long)  # TODO
            #
            results[code].append([features, adj, labels, idx_train, idx_val, idx_test])
        print("(%03d/%03d) Data of %s load done." % (count, num, str(code).ljust(5, ' ')))
    return results
def load_data_new(useACL=True):
    results = {}  # 用于保存返回结果
    # 加载数据集
    if useACL:
        # datasets = pd.read_csv('./data/ACL-V2.csv', dtype=np.str)  # TODO
        
        datasets = pd.read_csv('./data/ACL-V2-new.csv')
        
        # datasets = pd.read_csv('./data/ACL-V2-P.csv',index_col=0)  # TODO
        # datasets = pd.read_csv('./data/ACL-V2-P-30day.csv',index_col=0)  # TODO
        print('datasets.shape:{}'.format(datasets.shape))
    else:
        # datasets = pd.read_csv('./data/CSI500-SUB5-20190701-20200630-5M-Done.csv', dtype=np.str)
        
        # KDD2488 
        # datasets = pd.read_csv('./data/KDD-V2-new.csv')
        
        # KDD2487 from proprecessed data
        datasets = pd.read_csv('./data/KDD-V2-new_v3.csv')

        print('datasets.shape:{}'.format(datasets.shape))
    # 按 code 进行分组
    if useACL:
        by = 'Code'
    else:
        by = 'Code'
    count, num = 0, len(set(datasets[by]))
    for code, dataset in datasets.groupby(by=by):
        D_data = []
        X_locate = []
        Y_locate = []
        #
        # dataset = dataset.loc[('2014-01-01' <= dataset['Date']) & (dataset['Date'] <= '2016-01-01'), :]
        # for new dataset
        
        # dataset = dataset.loc[('2014/1/1' <= dataset['Date']) & (dataset['Date'] <= '2016/1/1'), :]
        
        if code in ['AGFS','BABA','GMRE']:
        # if code == 'GMRE':
            continue
        results[code] = []
        # 剔除含有 nan 的行
        dataset.dropna(axis=0, how='any', inplace=True)
        # print('dataset.shape:{}'.format(dataset.shape))
        # 对特征进行标准化
        if useACL:
            step = 504
            dataset.iloc[:, 2:13] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:13])  # TODO
            # dataset.iloc[:, 2:57] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:57])  # TODO
            # dataset.iloc[:, 2:112] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:112])  # TODO
            # dataset.iloc[:, 2:332] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:332])  # TODO

        else:
            # dataset.iloc[:, 3:44] = MinMaxScaler().fit_transform(dataset.iloc[:, 3:44])  # TODO
            # step = 2488
            step = 2487
            dataset.iloc[:, 2:13] = MinMaxScaler().fit_transform(dataset.iloc[:, 2:13])  # TODO
        # 剔除标签为 0 的数据
        dataset = dataset.loc[dataset['label'] != '0', :]

        # 分段截取数据
        # step = 2000
        # step = 504
        for index in range(0, dataset.shape[0], step):
            count += 1
            #
            data = dataset.iloc[index:index + step, :]
            # 对数据集进行切分
            date = data['Date'].values.tolist()
            get_train_size = False
            for i in range(len(date)):
                if not get_train_size and date[i] > '2015-08-01':
                    train_size = i
                    get_train_size = True
                elif get_train_size and date[i] > '2015-10-01':
                    val_size = i
                    break
            # idx_train, idx_val, idx_test = range(train_size), range(train_size, val_size), range(val_size, len(date))
            # idx_train, idx_val, idx_test = range(train_size), range(train_size, val_size), range(val_size, val_size+1)
            
            # idx_train, idx_val, idx_test = range(data.shape[0] - 120), range(data.shape[0] - 120, data.shape[0] - 60), range(data.shape[0] - 60, data.shape[0])
            
            # 6:2:2
            if useACL:
                idx_train, idx_val, idx_test = range(data.shape[0] - 202), range(data.shape[0] - 202, data.shape[0] - 101), range(data.shape[0] - 101, data.shape[0])
            else:

                idx_train, idx_val, idx_test = range(data.shape[0] - 474), range(data.shape[0] - 474, data.shape[0] - 222), range(data.shape[0] - 222, data.shape[0])
            # 7:1:2 err
            # idx_train, idx_val, idx_test = range(data.shape[0] - 253), range(data.shape[0] - 253, data.shape[0] - 101), range(data.shape[0] - 101, data.shape[0])
            
            #
            idx_features_labels = data.values
            idx_features_labels_data = idx_features_labels[:,2:13]
            # idx_features_labels_data = idx_features_labels[:,2:57]
            # idx_features_labels_data = idx_features_labels[:,2:112]
            idx_features_labels_data = idx_features_labels_data.tolist()
            # print('idx_f_labels_data.shape:{}'.format(len(idx_features_labels_data)))
            for i in range(len(idx_features_labels_data)):
                for j in range(len(idx_features_labels_data)):
                    np_i = np.array(idx_features_labels_data[i])
                    np_j = np.array(idx_features_labels_data[j])
                    dis = compute_Euclidean_Distance(np_i,np_j)
                    D_data.append(dis)
                    X_locate.append(i)
                    Y_locate.append(j)
            nodes_num = len(idx_features_labels_data)
            np_adj_data = np.array(D_data)
            np_row_locate = np.array(X_locate)  # coo_matrix 位置数据准备
            np_column_locate = np.array(Y_locate)  # coo_matrix 位置数据准备
            static_industry_adj = sp.coo_matrix((np_adj_data, (np_row_locate, np_column_locate)),
                                                shape=(nodes_num, nodes_num), dtype=np.float32)
            # adj = normalize_adj(static_industry_adj).toarray()
            adj = normalize_adj(static_industry_adj + sp.eye(static_industry_adj.shape[0]))
            # # 提取 idx
            # if useACL:
            #     idx = np.array(idx_features_labels[:, 0], dtype=np.str)
            # else:
            #     idx = np.array(idx_features_labels[:, 1], dtype=np.str)
            # idx_map = {j: i for i, j in enumerate(idx)}
            # 提取 features
            if useACL:
                features = sp.csr_matrix(idx_features_labels[:, 2:13], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:57], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:112], dtype=np.float)  # TODO
                # features = sp.csr_matrix(idx_features_labels[:, 2:332], dtype=np.float)  # TODO
            else:
                features = sp.csr_matrix(idx_features_labels[:, 3:44], dtype=np.float)
                # features = sp.csr_matrix(idx_features_labels[:, 2:57], dtype=np.float)  # TODO
            # 提取 labels
            labels = encode_onehot(idx_features_labels[:, -1])
            # 基于上述数据构建邻接矩阵
            edges_unordered = []
            # for _, group in data.iloc[0: train_size, :].groupby(by='label'):
            # # for _, group in data.groupby(by='label'):
            #     #
            #     if useACL:
            #         # edges_unordered.extend(list(itertools.product(group['Date'].values, repeat=2)))
            #         # edges_unordered.extend(list(itertools.combinations(group['Date'].values, 2)))
            #         edges_unordered.extend(list(itertools.combinations_with_replacement(group['Date'].values, 2)))
            #         # 只考虑 100 天以内的
            #         # date = group['Date'].values
            #         # for i in range(0, len(date)):
            #         #     edges_unordered.extend(list(itertools.combinations(date[i: i + 300], 2)))  # 不包含自己到自己的边
            #         # edges_unordered.extend(list(zip(date, date)))  # 自己到自己的边
            #     else:
            #         edges_unordered.extend(list(itertools.product(group['time'].values, repeat=2)))  # TODO
            # edges_unordered = np.array(edges_unordered)
            # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            #     edges_unordered.shape)
            # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            #                     shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
            # # 对邻接矩阵进行标准化
            # adj = normalize_adj(adj)  # 无需加上单位对角矩阵
            #
            features = torch.tensor(np.array(features.todense()), dtype=torch.float)  # TODO
            # print('features.shape:{}'.format(features.shape))
            adj = torch.tensor(np.array(adj.todense()), dtype=torch.float)  # TODO
            labels = torch.tensor(np.where(labels)[1], dtype=torch.long)  # TODO
            #
            idx_train = torch.tensor(idx_train, dtype=torch.long)  # TODO
            idx_val = torch.tensor(idx_val, dtype=torch.long)  # TODO
            idx_test = torch.tensor(idx_test, dtype=torch.long)  # TODO
            #
            results[code].append([features, adj, labels, idx_train, idx_val, idx_test])
        print("(%03d/%03d) Data of %s load done." % (count, num, str(code).ljust(5, ' ')))
    return results

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def compute_Euclidean_Distance(vector1,vector2):
    # op1 = np.sqrt(np.sum(np.square(vector1 - vector2)))
    op2 = np.linalg.norm(vector1 - vector2)
    return op2

def compute_evaluation_metrics(y_true, y_pred):
    y_pred = y_pred.max(1)[1].type_as(y_true)
    acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())
    f1 = f1_score(y_true.cpu(), y_pred.cpu())
    pre = precision_score(y_true.cpu(), y_pred.cpu())
    rec = recall_score(y_true.cpu(), y_pred.cpu())
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    mcc = matthews_corrcoef(y_true.cpu(), y_pred.cpu())
    return acc, f1, pre, rec, auc, mcc

