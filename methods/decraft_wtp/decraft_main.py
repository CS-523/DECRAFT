import os
import pickle

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from . import *


def add_parameters(global_model, client_model):
    for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
        server_param.data = (server_param.data + client_param.data.clone()) / 2
    return global_model


def update_model(global_model, client_model):
    for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
        client_param.data = server_param.data.clone()
    # return client_model


def check_para(model_list, w_ag, layer_num=1):
    para_list = []
    # layer_num = 2
    for tmp_num in range(len(model_list)):
        a = model_list[tmp_num].layers[layer_num].weight
        para_list.append(copy.deepcopy(a))
    b = w_ag.layers[layer_num].weight
    return para_list, b


def check_para_grad(model_list, w_ag, layer_num=1):
    para_list = []
    # layer_num = 2
    for tmp_num in range(len(model_list)):
        a = model_list[tmp_num].layers[layer_num].weight.grad
        para_list.append(copy.deepcopy(a))
    b = w_ag.layers[layer_num].weight.grad
    return para_list, b


def check_para_grad_single(model_list, layer_num=1):
    para_list = []
    # layer_num = 2
    for tmp_num in range(len(model_list)):
        a = model_list[tmp_num].layers[layer_num].weight.grad
        para_list.append(copy.deepcopy(a))
    # b = w_ag.layers[layer_num].weight.grad
    return para_list


# def FedAvg(w):
#     # time_start = time.time()
#     # mem_consumer = {}
#     w_avg = copy.deepcopy(w[0])
#     # for param in w_avg.parameters():
#     #     param.data.zero_()
#     # print(*[name for name, _ in w[0].named_parameters()], sep='\n')
#     for tmp_num in range(1, len(w)):
#         w_avg = add_parameters(w_avg, w[tmp_num])
#
#     # time_end = time.time()
#     # w_mem = get_memory_para(w_avg) * len(w) / 1024 / 1024
#     # mem_consumer['total_mem_cons'] = w_mem
#     # mem_consumer['total_time_cons'] = time_end - time_start
#     # return w_avg, mem_consumer
#     return w_avg

def add_parameters2(global_model, client_model, w=0.25):
    for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
        server_param.data += client_param.data.clone() * w
    return global_model


def FedAvg(w):
    # time_start = time.time()
    # mem_consumer = {}
    w_avg = copy.deepcopy(w[0])
    for param in w_avg.parameters():
        param.data.zero_()
    # print(*[name for name, _ in w[0].named_parameters()], sep='\n')
    weight = 1 / len(w)

    for tmp_num in range(len(w)):
        w_avg = add_parameters2(w_avg, w[tmp_num], weight)

    # time_end = time.time()
    # w_mem = get_memory_para(w_avg) * len(w) / 1024 / 1024
    # mem_consumer['total_mem_cons'] = w_mem
    # mem_consumer['total_time_cons'] = time_end - time_start
    # return w_avg, mem_consumer
    return w_avg


def decraft_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(
        device) for col in cat_features}
    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        model_build = ModelBuildingDecentralize()
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)
        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )

        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        #  global model
        model_1 = GraphAttnModel(in_feats=feat_df.shape[1],
                                 hidden_dim=args['hid_dim'] // 4,
                                 n_classes=2,
                                 heads=[4] * args['n_layers'],  # [4,4,4]
                                 activation=nn.PReLU(),
                                 n_layers=args['n_layers'],
                                 drop=args['dropout'],
                                 device=device,
                                 gated=args['gated'],
                                 ref_df=feat_df.iloc[train_idx],
                                 cat_features=cat_feat).to(device)
        print(model_1)
        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)  # 0.00075
        model_num = args['n_layers'] + 1
        model_list = model_build.build_node_model(feat_df=feat_df, args=args, device=device, train_idx=train_idx,
                                                  model_num=model_num)

        optimizer_list, lr_scheduler_list = model_build.build_node_optimizer(model_list=model_list, args=args, lr=lr)

        # optimizer = optim.Adam(model.parameters(), lr=lr,
        #                        weight_decay=args['wd'])
        # lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
        #     4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        # start_epoch, max_epochs = 0, 2000
        start_epoch = 0
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model_build.build_node_model_train(model_list=model_list)
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat,
                                                                                               labels,
                                                                                               seeds, input_nodes,
                                                                                               device)

                # (|input|, feat_dim); null; (|batch|,); (|input|,)
                blocks = [block.to(device) for block in blocks]

                train_batch_logits = model_build.model_farward(model_list=model_list, blocks=blocks,
                                                               lpa_labels=lpa_labels)
                # train_batch_logits.requires_grad = True
                # train_batch_logits.requires_grad_=True
                # train_batch_logits = model(
                #     blocks, batch_inputs, lpa_labels, batch_work_inputs)

                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0
                train_batch_logits.retain_grad()
                train_loss = loss_fn(train_batch_logits, batch_labels)

                # backward
                for tmp_optimizer_num in optimizer_list:
                    tmp_optimizer = optimizer_list[tmp_optimizer_num]
                    tmp_optimizer.zero_grad()
                # tmp_model = model_list[len(model_list) - 1]
                # print(tmp_model.layers[1].weight.grad)
                train_loss.backward()
                # tmp_model = model_list[len(model_list) - 1]
                # # print(tmp_model.layers[10][0].weight)
                # # print(tmp_model.layers[10][0].weight.grad)
                # print(tmp_model.layers[1].weight.grad)
                # tmp_model_2 = model_list[len(model_list) - 2]
                #
                # print(tmp_model_2.layers[1].weight.grad)
                # tmp_model_3 = model_list[len(model_list) - 3]
                #
                # print(tmp_model_3.layers[1].weight.grad)
                # tmp_model_4 = model_list[len(model_list) - 4]

                # print(tmp_model_4.layers[1].weight.grad)
                # print(tmp_model_4.layers[1].weight)
                # w_avg = FedAvg(model_list)

                # para_list_grad, w_a_grad = check_para_grad(model_list, w_avg)
                # para_list_grad = check_para_grad_single(model_list)
                for tmp_optimizer_num in optimizer_list:
                    tmp_optimizer = optimizer_list[tmp_optimizer_num]
                    tmp_optimizer.step()
                # print(tmp_model_4.layers[1].weight.grad)
                # print(tmp_model_4.layers[1].weight)
                for tmp_lr_scheduler_num in lr_scheduler_list:
                    tmp_lr_scheduler = lr_scheduler_list[tmp_lr_scheduler_num]
                    tmp_lr_scheduler.step()

                train_loss_list.append(train_loss.cpu().detach().numpy())
                w_avg = FedAvg(model_list)
                # para_list, w_a = check_para(model_list, w_avg)
                # para_list_grad, w_a_grad = check_para_grad(model_list, w_avg)
                # print(w_avg.layers[1].weight)
                for tmp_num in model_list:
                    # print(model_list[tmp_num].layers[0].weight)
                    update_model(w_avg, model_list[tmp_num])
                    # print(model_list[tmp_num].layers[0].weight)
                # model = add_parameters(model, w_avg)
                # para_list2, w_a2 = check_para(model_list, w_avg)
                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()

                    # if (len(np.unique(score)) == 1):
                    #     print("all same prediction!")
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(),
                                                                                               score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(
                                                                                               batch_labels.cpu().numpy(),
                                                                                               score)))
                    except:
                        pass

            # mini-batch for validation-
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            # model.eval()
            model_build.build_node_model_eval(model_list)
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat,
                                                                                                   labels,
                                                                                                   seeds, input_nodes,
                                                                                                   device)

                    blocks = [block.to(device) for block in blocks]
                    # val_batch_logits = model(
                    #     blocks, batch_inputs, lpa_labels, batch_work_inputs)
                    val_batch_logits = model_build.model_farward(model_list=model_list, blocks=blocks,
                                                                 lpa_labels=lpa_labels)

                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                                    loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                                   torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                                :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list / val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(),
                                                                                        score)))
                        except:
                            pass

            earlystoper.earlystop(val_loss_list / val_all_list, copy.deepcopy(w_avg))
            # earlystoper.earlystop(val_loss_list / val_all_list, w_avg)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = NodeDataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model_test = earlystoper.best_model.to(device)
        # update the global model
        model_list_test = copy.deepcopy(model_list)
        for tmp_num in model_list_test:
            update_model(b_model_test, model_list_test[tmp_num])
        # b_model.eval()
        model_build.build_node_model_eval(model_list_test)

        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat,
                                                                                               labels,
                                                                                               seeds, input_nodes,
                                                                                               device)

                blocks = [block.to(device) for block in blocks]
                # test_batch_logits = b_model(
                #     blocks, batch_inputs, lpa_labels, batch_work_inputs)
                test_batch_logits = model_build.model_farward(model_list=model_list_test, blocks=blocks,
                                                              lpa_labels=lpa_labels)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(
                    test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2  # flag list
    y_target[mask] = 0  #
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))


def load_decraft_data(dataset: str, test_size: float):
    """
    Load graph, feature, and label given dataset name - decentralized manner
    :param dataset: the dataset name
    :param test_size: the size of test set
    :returns: feature, label, graph, category features
    """
    # prefix = './antifraud/data/'

    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]
        ###
        feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)
        ###
        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size / 2,
                                                                random_state=2, shuffle=True)

    elif dataset == "elliptic":
        cat_features = []

        datadir_path = os.path.join(prefix, 'elliptic_bitcoin_dataset')
        classes_csv = os.path.join(datadir_path, 'elliptic_txs_classes.csv')
        edgelist_csv = os.path.join(datadir_path, 'elliptic_txs_edgelist.csv')
        features_csv = os.path.join(datadir_path, 'elliptic_txs_features.csv')
        # classes: (203769, 1);

        classes = pd.read_csv(classes_csv)  # labels are 'unknown', '1'(illicit), '2'(licit)
        classes['class'] = classes['class'].map({'unknown': 2, '1': 1, '2': 0})
        edgelist = pd.read_csv(edgelist_csv)

        features = pd.read_csv(features_csv, header=None)  # features of the transactions: (203769, 167)
        # data = pd.concat([classes, features], axis=1)
        transaction_id_map = dict(
            zip(features[0].values, features.index.values))  # {230425980: 0, 5530458: 1,}
        # transaction_id = np.unique(features[0].values)  # 203769
        # feature_idx = [i + 2 for i in range(93 + 72)]
        feat_data = features.drop(columns=[0, 1])
        #     feat_data, labels, train_idx, test_idx, g, cat_features
        labels = classes['class']
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2,
                                                                shuffle=True)
        edgelist['txId1'] = edgelist['txId1'].map(transaction_id_map)
        edgelist['txId2'] = edgelist['txId2'].map(transaction_id_map)
        src = edgelist['txId1'].values
        tgt = edgelist['txId2'].values
        g = dgl.graph((src, tgt))
        # g = dgl.DGLGraph(multigraph=True)
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        # degrees = g.in_degrees()
        # out_degrees = g.out_degrees()
        g = dgl.add_self_loop(g)
        graph_path = prefix + "graph-{}-decraft.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])


    elif dataset == "bitcoin":
        pass

    elif dataset == 'phishing':

        cat_features = []
        datadir_path = os.path.join(prefix, 'Phishing')

        def load_pickle(fname):
            with open(fname, 'rb') as f:
                return pickle.load(f)

        datadir_file_path = os.path.join(datadir_path, 'MulDiGraph.pkl')
        G = load_pickle(datadir_file_path)
        print(nx.info(G))

    elif dataset == "yelp":
        cat_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists--------yelp_homo = yelp['homo']
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))  # 45954 yelp
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2,
                                                                shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)  # 1193186 yelp
        tgt = np.array(tgt)  # 1193186 yelp
        g = dgl.graph((src, tgt))

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        cat_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
    return feat_data, labels, train_idx, test_idx, g, cat_features
