from __future__ import division
from __future__ import print_function

import os
import glob
import time
import torch
import random
import logging
import datetime
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

from models import GAT, SpGAT
from utils import load_data_new, compute_accuracy, compute_evaluation_metrics

warnings.filterwarnings('ignore')

filepath = os.path.join(os.path.split(__file__)[0], 'logs', datetime.datetime.today().strftime('%Y %m %d') + ' ' +
                        os.path.split(__file__)[-1].split('.')[0] + '.log')

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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# remove seed
'''
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
'''
cuda_device = 0

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    t = time.time()
    # model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # 随机从idx_train里面进行不放回抽样batch_size 个，形成新的idx_train
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = compute_accuracy(output[idx_train], labels[idx_train])
    # acc_train, f1_train, pre_train, rec_train, auc_train, mcc_train = compute_evaluation_metrics(labels[idx_train],output[idx_train])

    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        model.eval()
        output = model(features, adj)
    #
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = compute_accuracy(output[idx_val], labels[idx_val])
    logger.debug('Epoch(%04d) ===> loss_train: %.4f acc_train: %.4f loss_val: %.4f acc_val: %.4f time: %.4f' % (
        epoch + 1, loss_train, acc_train, loss_val.data.item(), acc_val.data.item(),
        time.time() - t))
    ''' 
    logger.debug('Epoch(%04d) ===> loss_train: %.4f acc_train: %.4f loss_val: %.4f acc_val: %.4f time: %.4f' % (
        epoch + 1, loss_train.data.item(), acc_train.data.item(), loss_val.data.item(), acc_val.data.item(),
        time.time() - t))
    '''
    '''
    logger.debug('Epoch(%04d) ===> loss_train: %.4f acc_train: %.4f loss_train: %.4f acc_train: %.4f time: %.4f' % (
        epoch + 1, loss_train.data.item(), acc_train.data.item(), loss_train.data.item(), acc_train.data.item(),
        time.time() - t))
    '''
    model.train()
    return loss_val.data.item()

    # return loss_train.data.item()


def compute_test(count, codes, best_epoch):
    model.eval()
    output = model(features, adj)
    # "save the output for GAT"
    # "add code"
    test_length = len(output[idx_test])
    code_list = []
    for i in range(test_length):
        code_list.append(code)
    output_save = output[idx_test].cpu().detach().numpy()
    # row_sums = output_save.sum(axis=1)
    # new_output_save = output_save / row_sums[:, np.newaxis]
    # new_output_save = np.around(new_output_save, 4)
    pd_new_output_save = pd.DataFrame(output_save,
                                      columns=['gatclass_1','gatclass_2'])
    pd_new_output_save['Code'] = code_list
    #
    pd_new_output_save.to_csv('./gatoutput/gatoutput.csv',header=False,mode='a')
    print(code,'write success')
    # # save the gat output for analysing
    # output_gat = torch.from_numpy(new_output_save)

    # "get output from lstm. pretrain lstm model and save results"
    # output_data = pd.read_csv('./gatoutput/lstmoutput_test_4.csv', header=None, delimiter=',')
    # # print(output_data)
    #
    # output_data.columns = ['id', 'lstm_class_1', 'lstm_class_2', 'Code', 'val_acc']
    # target_stock_data = output_data[['lstm_class_1', 'lstm_class_2']][output_data['Code'] == code]

    # "add lstm val acc for α judging"
    # val_acc_list = output_data['val_acc'][output_data['Code'] == code].values.tolist()
    # target_stock_data = np.array(target_stock_data)
    # output_lstm = torch.from_numpy(target_stock_data)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])

    # "original acc_test for gat output"
    # acc_test = compute_accuracy(output[idx_test], labels[idx_test])
    
    acc_test, f1_test, pre_test, rec_test, auc_test, mcc_test = compute_evaluation_metrics(labels[idx_test],output[idx_test])
    
    # preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
    # print('preds:{}'.format(preds))
    #
    # "compute acc_test for complex output from lstm and gat"
    # "take the averaging voting method for model choice"
    # alpha_list = []
    # for val_acc_value in val_acc_list:
    #     if val_acc_value < 0.5:
    #         alpha = 0.7
    #     else:
    #         alpha = 0.3
    #     alpha_list.append(alpha)
    # alpha_value = torch.Tensor(alpha_list)
    # alpha_matrix = torch.cat([alpha_value, alpha_value])
    # alpha_matrix = torch.reshape(alpha_matrix, [2, -1]).T
    # output_gat = output[idx_test].cpu()  # move to the same device because lstm output in cpu
    # acc_test = accuracy(
    #     (torch.mul(output_lstm, (torch.ones_like(output_lstm) - alpha_matrix)) + torch.mul(output_gat, alpha_matrix)),
    #     labels[idx_test])
    # # single alpha value test
    # # acc_test = accuracy((output_lstm * (1-alpha) + output_gat * alpha) ,labels[idx_test])
    # logger.info('(%03d/%03d) Loading %04d th epoch ===> loss_test: %.4f acc_test: %.4f' % (
    #     count, codes, best_epoch, loss_test.item(), acc_test.item()))
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info('(%03d/%03d) Loading %04d th epoch ===> loss_test: %.4f acc_test: %.4f' % (
        count, codes, best_epoch, loss_test.item(), acc_test.item()))
    # original return
    # return acc_test.item()
    return acc_test, f1_test, pre_test, rec_test, auc_test, mcc_test


# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# results = load_data()


# for ACL18
# results = load_data_new()
# for KDD17
results = load_data_new(useACL=False)
# for i in range(3):  # TODO

accuracy_record, count, codes = [], 0, len(results.keys())
f1_record, pre_record, rec_record, auc_record, mcc_record = [], [], [], [], []
for code, result in results.items():
    #
    accuracy_list, count = [], count + 1
    #
    logger.info("Code: {}".format(code))
    #
    if args.sparse:
        # model = SpGAT(nfeat=41, nhid=args.hidden, nclass=3, dropout=args.dropout, nheads=args.nb_heads, alpha=args.alpha)  # TODO
        model = SpGAT(nfeat=6, nhid=args.hidden, nclass=3, dropout=args.dropout, nheads=args.nb_heads,
                      alpha=args.alpha)
    else:
        model = GAT(nfeat=11, nhid=args.hidden, nclass=2, dropout=args.dropout, nheads=args.nb_heads,
                    alpha=args.alpha)  # TODO==
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #
    for features, adj, labels, idx_train, idx_val, idx_test in result:
        # print(features)
        if args.cuda:
            model.cuda(cuda_device)

            features = features.cuda(cuda_device)
            adj = adj.cuda(cuda_device)
            labels = labels.cuda(cuda_device)

            idx_train = idx_train.cuda(cuda_device)
            idx_val = idx_val.cuda(cuda_device)
            idx_test = idx_test.cuda(cuda_device)

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)

        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = args.epochs + 1
        best_epoch = 0
        for epoch in range(args.epochs):

            loss_values.append(train(epoch))

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))

            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    try:
                        os.remove(file)
                    except Exception as error:
                        os.system("del /f /q %s" % file)
                    # os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                try:
                    os.remove(file)
                except Exception as error:
                    os.system("del /f /q %s" % file)
                # os.remove(file)

        logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        # Testing
        # acc_test = compute_test(count, codes, best_epoch + 1)
        
        acc_test, f1_test, pre_test, rec_test, auc_test, mcc_test = compute_test(count, codes, best_epoch + 1)

        accuracy_list.append(float('%.4f' % acc_test))
        accuracy_record.append(float('%.4f' % acc_test))
        f1_record.append(float('%.4f' % f1_test))
        pre_record.append(float('%.4f' % pre_test))
        rec_record.append(float('%.4f' % rec_test))
        auc_record.append(float('%.4f' % auc_test))
        mcc_record.append(float('%.4f' % mcc_test))
        torch.cuda.empty_cache()

logger.info("Mean accuracy: %.4f f1: %.4f pre: %.4f rec: %.4f auc: %.4f mcc: %.4f" % (float(np.mean(accuracy_record)), float(np.mean(f1_record)), float(np.mean(pre_record)), float(np.mean(rec_record)), float(np.mean(auc_test)), float(np.mean(mcc_test))))
