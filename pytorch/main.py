#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    writer = SummaryWriter('')
    # train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=2, shuffle=True, drop_last=True)
    #print("data loader")
    #print("train loader length", len(train_loader))
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(ModelNet40(partition='val', num_points=args.num_points), num_workers=8,
                             batch_size=2, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
        #model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    #criterion = cal_loss
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    criterion = nn.CrossEntropyLoss()
    best_test_loss = float('inf')
    best_test_acc = 0
    count = 0
    df = pd.DataFrame()
    df_label = pd.DataFrame()
    df_data = pd.DataFrame()
    #for epoch in range(args.epochs):
    for epoch in range(5):
        print("epoch", epoch)
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            label = torch.tensor(label, dtype=torch.long, device=device)
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            #print("data size", data.size())
            #print("label size", label.size())
            #print(label[0][0])
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            #print("logits", logits)
            #print("logits size", logits.size())
            logits = logits.to(device)
            #print(logits[0][0])
            #print("logits size", logits.size())
            #logits = logits.permute(0, 2, 1).contiguous()
            #print("permuted logits size", logits.size())
            #num_vertices = 961
            #atch_size = 2
            #num_labels = 75
            #labels = torch.empty(batch_size, num_vertices, dtype=torch.long).random_(num_labels)
            #labels = labels.to(device)
            #preds = torch.randn(batch_size, num_labels, num_vertices, requires_grad=True)
            #print("preds size", preds.size())
            #print(preds)
            #y_tensor = torch.tensor(logits, dtype=torch.long, device=device)
            #print("preds size", preds.size())
            #print("label size", label.size())
            #print("fake label size", labels.size())
            loss = criterion(logits, label)
            #loss = criterion(preds, labels)
            #loss.backward()
            #opt.step()
            train_loss += loss.item() * 2
    #         preds = logits.max(dim=1)[1]
            count += batch_size
    #         train_loss += loss.item() * batch_size
    #         train_true.append(label.cpu().numpy())
    #         train_pred.append(preds.detach().cpu().numpy())
    #     train_true = np.concatenate(train_true)
    #     train_pred = np.concatenate(train_pred)
        print('train loss', train_loss * 1.0 / count)
    #     writer.add_scalar('training loss', train_loss / count, epoch)
    #     ####################
    #     # Test
    #     ####################
    #     test_loss = 0.0
    #     count = 0.0
    #     model.eval()
    #     test_pred = []
    #     test_true = []
    #     for data, label in test_loader:
    #         data, label = data.to(device), label.to(device)
    #         data = data.permute(0, 2, 1)
    #         batch_size = data.size()[0]
    #         #print("data length", len(data))
    #         #print("data 0 length", len(data[0]))
    #         #print("data 0 0 length", len(data[0][0]))
    #         #print("data 0 1 length", len(data[0][1]))
    #         #print("data 0 2 length", len(data[0][2]))
    #         logits = model(data)
    #         logits = logits.to(device)
    #         #print("logits length", len(logits))
    #         #print("test predictions", logits)
    #         loss = criterion(logits, label)
    #         preds = logits.max(dim=1)[1]
    #         count += batch_size
    #         test_loss += loss.item() * batch_size
    #         test_true.append(label.cpu().numpy())
    #         test_pred.append(preds.detach().cpu().numpy())
    #     test_true = np.concatenate(test_true)
    #     test_pred = np.concatenate(test_pred)
    #     print('test loss', test_loss * 1.0 / count)
    #     #df[str(epoch)] = [float(x) for x in logits[0]]
    #     #df_label[str(epoch)] = [float(x) for x in label[0]]
    #     #print("data", data.tolist()[0])
    #     # data_list = []
    #     # for x, y, z in zip(data.tolist()[0][0], data.tolist()[0][1], data.tolist()[0][2]):
    #     #     data_list.append(x)
    #     #     data_list.append(y)
    #     #     data_list.append(z)
    #     # df_data[str(epoch)] = data_list
    #     #print("data length", len(data.tolist()))
    #     #print("data 0 length", len(data.tolist()[0]))
    #     #print("data 0 0 length", len(data.tolist()[0][0]))
    #     writer.add_scalar('validation loss', test_loss / count, epoch)
    #     if test_loss <= best_test_loss:
    #         best_test_loss = test_loss
    #         torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
    # writer.close()
    #df.to_csv('val_predictions_simplified.csv')
    #df_label.to_csv('val_labels_simplified.csv')
    #df_data.to_csv('data_simplified.csv')


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)