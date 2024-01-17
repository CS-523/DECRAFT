#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Traceless'
__mtime__ = '2023/12/6'
"""
import math
import os.path
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
# 使用同步训练所需的时间
# 第一位是t_u，第二维是t_d
import pandas as pd
import sympy as sp


class layered_node:
    def __init__(self):
        self.leader_node = 0
        self.orid_node_flag = []
        self.leaf_flag = False  # 当前是否属于最后一层
        self.ascend_raw_data = []
        self.time_con = 0

    def update_layer_content(self, leader_node, leaf_flag, ascend_raw_data):
        self.leader_node = leader_node
        self.leaf_flag = leaf_flag
        self.ascend_raw_data = ascend_raw_data

    def update_time_con(self, timecon):
        self.time_con = timecon


def obtain_raw_time(raw_data):
    # 只有一个聚合节点时
    # 用于处理极端的情况的
    time_con = np.sum(raw_data, axis=1)
    max_time_con = np.max(time_con)

    t_u_0 = np.min(raw_data[:, 0])
    max_time_con2 = t_u_0 + np.sum(raw_data[:, 1])

    final_time = np.max([max_time_con, max_time_con2])

    return final_time


def obtain_precision_raw_time(raw_data):
    # 只有一个聚合节点时，由模型更新最快的节点来聚合整个模型
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    t_u_0 = ascend_raw_data[0, 0]
    current_time = t_u_0
    t_u_max = ascend_raw_data[-1, 0]
    finish_update_node = 0
    for i in range(1, ascend_raw_data.shape[0]):
        if current_time < t_u_max:

            if current_time < ascend_raw_data[i, 0]:
                # 说明当前时间，该节点还没有完成了更新，需要等待节点更新模型之后启动上传
                current_time = ascend_raw_data[i, 0]
            current_time += ascend_raw_data[i, 1]  # 启动上传工作、
        else:
            finish_update_node = i
            break

    time_con = np.sum(raw_data[finish_update_node:, 1])
    current_time += time_con
    # # 用于处理极端的情况的
    # time_con = np.sum(raw_data, axis=1)
    # max_time_con = np.max(time_con)
    #
    # t_u_0 = np.min(raw_data[:, 0])
    # max_time_con2 = t_u_0 + np.sum(raw_data[:, 1])
    #
    # final_time = np.max([max_time_con, max_time_con2])
    return current_time


def obtain_k_syn_method(raw_data):
    # 采用穷举的方法，获得每一个K值所代表的时间长短
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    # ascend_raw_data = np.sort(raw_data, axis=0)
    tmp_time_list = []
    E_t_d = np.mean(raw_data[:, 1])
    max_t_u = np.max(raw_data[:, 0])
    n_c = raw_data.shape[0]
    for k in range(1, ascend_raw_data.shape[0]):
        a = (n_c - k) / k
        tmp_max_t_d = np.max(ascend_raw_data[:k, 1])
        tmp_max_t_u = np.max(ascend_raw_data[:k, 0])
        tmp_time = tmp_max_t_u + a * E_t_d + (k - 1) * tmp_max_t_d
        tmp_time_list.append(tmp_time)
    optimal_k = np.argmin(tmp_time_list) + 1
    print('optimal_k is {}'.format(optimal_k))
    print('optimal time consumption is {}'.format(tmp_time_list[optimal_k - 1]))
    return tmp_time_list


def two_adapt_time_con(k, ascend_raw_data):
    """
    根据k值分配不同的节点组合
    :param k:
    :param ascend_raw_data:
    :return:
    """
    # k = 3
    # allocation_lsit = []
    n_c = ascend_raw_data.shape[0]
    a = int(np.ceil((n_c - k) / k))
    index_list = np.arange(ascend_raw_data.shape[0])
    oridinary_index_list = index_list[k:]
    #  切分数据：https://blog.csdn.net/mingyuli/article/details/81227629
    np.random.shuffle(oridinary_index_list)
    split_slice_list = [i * a for i in range(1, k)]
    split_array = np.split(oridinary_index_list, split_slice_list)
    for i in range(k):
        split_array[i] = np.append(split_array[i], i)
    time_list = []  # 新的raw_data的训练时间
    for i in range(k):
        # 1、取出原始数据；
        # 2、计算每一个的时间消耗；
        cur_split_array = split_array[i]
        cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
        tmp_raw_data = ascend_raw_data[cur_flag_list]
        # oridinary_index_list: np.array
        cur_time_con = obtain_precision_raw_time(tmp_raw_data)
        time_list.append(cur_time_con)
        print('k: {}, time_con: {}'.format(k, cur_time_con))
    # first_stage_time = np.max(time_list)  # 新的raw_data的训练时间
    # 将 每一组当做一个节点，重新组合成一个新的raw_data
    new_raw_data = np.array([time_list, ascend_raw_data[:k, 1].tolist()]).T
    second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


def branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=False):
    """
    为了计算中间层外加叶子层中，每一层的时间消耗
    :param ascend_raw_data:
    :param cur_layer_flag_list:
    :param leader_layer_flag: 当前层是否为第一层，因为第一层需要去中心化的模型聚合，计算方式有所不一样的
    :return:
    """
    cur_time_con_list = []  # 当前层的时间消耗
    leader_node_raw_data = []  # 存储头结点的原始数据

    for tmp_raw_flag in cur_layer_flag_list:

        tmp_raw_data = ascend_raw_data[tmp_raw_flag]
        leader_node_raw_data.append(tmp_raw_data[0, :].tolist())

        tmp_hier_flag, tmp_cur_layer_flag_list, tmp_cur_time_con = layer_allocation(tmp_raw_data)

        tmp_n_c = tmp_raw_data.shape[0]
        print('hier_flag: {}, nc: {}, k: {}'.format(tmp_hier_flag, tmp_n_c, len(tmp_cur_layer_flag_list)))

        if tmp_hier_flag:  # 如果当前节点数据继续分层，会继续调用当前函数
            tmp_cur_time_con = branch_layer_time_con(tmp_raw_data, tmp_cur_layer_flag_list)
        cur_time_con_list.append(tmp_cur_time_con)  # 存储每一个簇中模型聚合所消耗的时间
    if leader_layer_flag:
        # TODO 应该修改为去中心化的方式，但是差别不是很大，所以暂时没有更改
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
        # final_time = second_stage_time
    else:
        # test_data = leader_node_raw_data[:, 1]  # list中是不能这样表示的
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


# def layered_allocation(ascend_raw_data):
#     hier_flag, cur_layer_flag_list, cur_time_con = layer_allocation(ascend_raw_data)  # 第一层，需要去中心化模型聚合的
#     # root_node = layered_node()
#     # root_node.update_layer_content(leader_node='root', leaf_flag=not hier_flag, )
#     # 应该按照某一支不断迭代，直到得到当前分支节点的时间消耗
#     for tmp_raw_flag in cur_layer_flag_list:
#         tmp_raw_data = ascend_raw_data[tmp_raw_flag]
#         # todo 更新当前所在分支的时间消耗--还是需要递归算法的啊
#         # hier_flag, cur_layer_flag_list, cur_time_con = layer_allocation(ascend_raw_data)
#     # todo 去中心化的模型聚合
#     pass


def layer_allocation(ascend_raw_data):
    """
    根据ascend_raw_data将当前数据进行分层
    :param ascend_raw_data:
    :return:
    """
    k1, k2 = obtain_adapt_syn_method(ascend_raw_data)

    hier_flag = check_layer_callback_k(k1, k2)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    cur_time_con = None
    cur_layer_flag_list = []
    if hier_flag:
        k = k2
        n_c = ascend_raw_data.shape[0]
        a = int(np.ceil((n_c - k) / k))
        index_list = np.arange(ascend_raw_data.shape[0])
        oridinary_index_list = index_list[k:]
        #  切分数据：https://blog.csdn.net/mingyuli/article/details/81227629
        np.random.shuffle(oridinary_index_list)
        split_slice_list = [i * a for i in range(1, k)]
        split_array = np.split(oridinary_index_list, split_slice_list)
        for i in range(k):
            split_array[i] = np.append(split_array[i], i)
        # time_list = []  # 新的raw_data的训练时间
        for i in range(k):
            # 1、取出原始数据；
            # 2、计算每一个的时间消耗；
            cur_split_array = split_array[i]
            cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
            cur_layer_flag_list.append(cur_flag_list)
            # tmp_raw_data = ascend_raw_data[cur_flag_list]
            # # oridinary_index_list: np.array
            # cur_time_con = obtain_precision_raw_time(tmp_raw_data)
            # time_list.append(cur_time_con)
            # print('k: {}, time_con: {}'.format(k, cur_time_con))
    else:
        # 说明当前已经是叶子节点，并不需要进一步往下分了哦
        cur_time_con = obtain_precision_raw_time(ascend_raw_data)

    return hier_flag, cur_layer_flag_list, cur_time_con


def layered_adapt_time_con(ascend_raw_data):
    """
    根据k值分配不同的节点组合
    :param k:
    :param ascend_raw_data:
    :return:
    """
    # k = 3
    # allocation_lsit = []
    n_c = ascend_raw_data.shape[0]
    hier_flag, cur_layer_flag_list, cur_time_con = layer_allocation(ascend_raw_data)  # 第一层，需要去中心化模型聚合的
    if hier_flag:
        final_time = branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=True)
    else:
        final_time = cur_time_con
    return final_time


def obtain_k_precision_syn_method(raw_data):
    # 采用穷举的方法，获得每一个K值所代表的时间长短
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    # ascend_raw_data = np.sort(raw_data, axis=0)
    tmp_time_list = []
    E_t_d = np.mean(raw_data[:, 1])
    max_t_u = np.max(raw_data[:, 0])
    n_c = raw_data.shape[0]
    for k in range(1, ascend_raw_data.shape[0]):
        # 根据k分配节点
        # node_allocation(k, ascend_raw_data)
        temp_time_con = two_adapt_time_con(k, ascend_raw_data)
        tmp_time_list.append(temp_time_con)
    optimal_k = np.argmin(tmp_time_list) + 1
    print('optimal_k is {}'.format(optimal_k))
    print('optimal time consumption is {}'.format(tmp_time_list[optimal_k - 1]))
    return tmp_time_list


def obtain_k_single_precision_syn_method(raw_data, k):
    # 采用穷举的方法，获得每一个K值所代表的时间长短
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    # ascend_raw_data = np.sort(raw_data, axis=0)
    tmp_time_list = []
    # 根据k分配节点
    # node_allocation(k, ascend_raw_data)
    temp_time_con = two_adapt_time_con(k, ascend_raw_data)
    # tmp_time_list.append(temp_time_con)
    return temp_time_con


def obtain_layered_precision_syn_method(raw_data):
    """
    获得分层之后的时间消耗
    :param raw_data: 原始数据
    :return:
    """

    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    # ascend_raw_data = np.sort(raw_data, axis=0)
    # 根据原始数据分配数据分配节点

    # node_allocation(k, ascend_raw_data)
    temp_time_con = layered_adapt_time_con(ascend_raw_data)
    return temp_time_con

# TODO 不需要再sort了呀，可以添加一个sort flag
def obtain_adapt_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    k_ = np.argmax(ascend_raw_data[:, 1])
    n_c = raw_data.shape[0]
    if n_c == 1:  # 只有一个数据就没有必要再往下分了哦
        return 0, 0
    E_t_d = np.mean(raw_data[:, 1])
    max_t_d = np.max(ascend_raw_data[:, 1])
    min_t_d = np.min(ascend_raw_data[:, 1])
    max_t_u = np.max(ascend_raw_data[:, 0])
    min_t_u = np.min(ascend_raw_data[:, 0])
    # 当k>K_
    k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u)), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # if k1 < k_:
    #     k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u + (k_-1)*(max_t_d-min_t_d))), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # 使用这种方式，有可能导致k1>2
    # k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + (1 - k_ / n_c) * max_t_u)), np.sqrt(n_c * E_t_d / max_t_d)

    print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))

    if np.ceil(k1) < k_:
        # 这种使得左下界的值变大了，有可能高于左上界
        # temp_k1 = n_c * n_c * E_t_d / (k_ * max_t_d + (k_ * max_t_u - n_c * min_t_u) + (k_ - 1) * (
        #             k_ * max_t_d - n_c * min_t_d))
        temp_k1 = n_c * E_t_d / (max_t_d + (max_t_u - n_c * min_t_u) + (k_ - 1) * (
                max_t_d - n_c * min_t_d))
        if temp_k1 > 0:
            k1 = np.sqrt(temp_k1)
        else:
            # 测试时使用
            # print(k1)
            # temp_k = np.sqrt(temp_k1)
            # test_nan = float('nan')
            # if math.isnan(temp_k):
            #     print('error')
            # if np.isnan(temp_k):
            #     print('error')
            # if math.isnan(test_nan):
            #     print('error')
            # if np.isnan(test_nan):
            #     print('error')
            k1 = 1
        k2 = np.sqrt(n_c * n_c * E_t_d / (max_t_d * k_))
        print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # 当通信时间消耗特别大的时候，又该如何嘞
    if np.isnan(k1) or k1 > k2:
        print('error')
        print(k1)
    x = sp.Symbol('x', positive=True)
    # f = 2*n_c/(x**3) + max_t_d - min_t_d
    f_test = max_t_u - min_t_u - n_c / (1 ** 2) * E_t_d + max_t_d + (1 - 1) * (max_t_d - min_t_d)
    print('f_test: {}'.format(f_test))
    f = max_t_u - min_t_u - n_c / (x ** 2) * E_t_d + max_t_d + (x - 1) * (max_t_d - min_t_d)
    # sp.solve(f)
    # 4是猜测的值
    x_test = sp.nsolve(f, x, k1, prec=3, verify=False)
    # if x_test > k1:
    if not math.isnan(x_test):
        k1 = x_test
    print('x_test: {}'.format(x_test))
    f_test = max_t_u - min_t_u - n_c / (x_test ** 2) * E_t_d + max_t_d + (x_test - 1) * (max_t_d - min_t_d)
    print('f_test2: {}'.format(f_test))
    # 求f的倒数
    k_1 = 1
    f_2_k = np.max(ascend_raw_data[:k_1, 1])
    f_1_k_ = 0 if ascend_raw_data[k_1 - 1, 0] > ascend_raw_data[k_1, 0] else ascend_raw_data[k_1, 0] - ascend_raw_data[
        k_1 - 1, 0]
    f_2_k_ = 0 if ascend_raw_data[k_1 - 1, 1] > ascend_raw_data[k_1, 1] else ascend_raw_data[k_1, 1] - ascend_raw_data[
        k_1 - 1, 1]
    f_1 = f_1_k_ - n_c / (k_1 ** 2) * E_t_d + f_2_k + (k1 - 1) * f_2_k_
    # k__ = 1
    print('f_1: {}'.format(f_1))
    print('finally: k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))

    # return int(np.floor(k1)), int(np.floor(k2))
    return k1, k2


def check_layer_callback(k1, k2):
    """

    :param k1: 上界
    :param k2: 下界
    :return: 是否继续分层
    """
    if np.ceil(k2) - np.ceil(k1) >= 1:
        print(True)
        hier_flag = True

    else:
        print(False)
        hier_flag = False

    return hier_flag


def check_layer_callback_k(k1, k2):
    """
    :param k2: 下届
    :return: 是否继续分层
    """
    if np.floor(k2) >= 2 and np.ceil(k2) - np.ceil(k1) >= 1:
        print(True)
        hier_flag = True
    else:
        print(False)
        hier_flag = False

    # if np.ceil(k2) - np.ceil(k1) >= 1:
    #     print(True)
    #     hier_flag = True
    #
    # else:
    #     print(False)
    #     hier_flag = False

    return hier_flag


def test_check_hier(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: 客户端数量
    :param plot_flag: 是否打印图片
    :return:
        hier_flag：是否分层
    """
    # TODO待做，这个条件下，不同客户端下面成功的个数
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # 生成0-1的随机数
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_raw_time(raw_data)
    # print('Single aggregation node time is {}'.format(final_time_1))
    # 根据不同的k值获得不同的时间消耗
    # tmp_time_list = obtain_k_syn_method(raw_data)
    k1, k2 = obtain_adapt_syn_method(raw_data)

    hier_flag = check_layer_callback_k(k1, k2)

    return hier_flag


def test_check_two_layer_adaptive_precision(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """
    :param data_num: 客户端数量
    :param plot_flag: 是否打印图片
    :return:
        hier_flag：是否分层
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # 根据不同的k值获得不同的时间消耗
    ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准
    ori_tmp_time_list_precision = obtain_k_precision_syn_method(raw_data)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    k1, k2 = obtain_adapt_syn_method(raw_data)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    if plot_flag:
        plt.plot(range(1, len(ori_tmp_time_list) + 1), ori_tmp_time_list, label='ori_ALL')
        plt.plot(range(1, len(ori_tmp_time_list_precision) + 1), ori_tmp_time_list_precision, label='ALL')
        plt.plot(range(k1, k2), ori_tmp_time_list[k1 - 1:k2 - 1], label='Adaption', marker='x', color='r')
        # plt.plot(final_time_1)
        plt.hlines(final_time_1, color='k', xmin=1, xmax=len(ori_tmp_time_list) + 1, label='Origin')
        plt.legend()
        plt.show()
    return ori_tmp_time_list, [k1, k2]


def test_check_layered_adaptive(data_num=1000, dist_flag=None, dis_para=None):
    """

    :param data_num: 客户端数量
    :param plot_flag: 是否打印图片
    :return:
        hier_flag：是否分层
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_precision_raw_time(raw_data)
    final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # 计算得到最优的k值
    k1, k2 = obtain_adapt_syn_method(raw_data)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准，但比较快
    two_layer_tmp_time = obtain_k_single_precision_syn_method(raw_data, k2)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    layered_time_con = obtain_layered_precision_syn_method(raw_data)

    return final_time_1, two_layer_tmp_time, layered_time_con


def test_check_two_layer_adaptive(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: 客户端数量
    :param plot_flag: 是否打印图片
    :return:
        hier_flag：是否分层
    """
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # 生成0-1的随机数
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # 根据不同的k值获得不同的时间消耗
    # ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准, 但比较快
    k1, k2 = obtain_adapt_syn_method(raw_data)
    # if np.ceil(k2) - np.ceil(k1) >= 1:
    #     print(True)
    #     hier_flag = True
    #
    # else:
    #     print(False)
    #     hier_flag = False
    # hier_flag = check_layer_callback_k(k1, k2)
    # print(tmp_time_list)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    if plot_flag:
        plt.plot(range(1, len(ori_tmp_time_list) + 1), ori_tmp_time_list, label='ori_ALL')
        # plt.plot(range(1, len(ori_tmp_time_list) + 1), ori_tmp_time_list, label='ALL')
        plt.plot(range(k1, k2), ori_tmp_time_list[k1 - 1:k2 - 1], label='Adaption', marker='x', color='r')
        # plt.plot(final_time_1)
        plt.hlines(final_time_1, color='k', xmin=1, xmax=len(ori_tmp_time_list) + 1, label='Origin')
        plt.legend()
        plt.show()
    return ori_tmp_time_list, [k1, k2]


def test_layered_adaptive_time(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: 客户端数量
    :param plot_flag: 是否打印图片
    :return:
        hier_flag：是否分层
    """

    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # 根据不同的k值获得不同的时间消耗
    ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准, 但比较快
    k1, k2 = obtain_adapt_syn_method(raw_data)
    # if np.ceil(k2) - np.ceil(k1) >= 1:
    #     print(True)
    #     hier_flag = True
    #
    # else:
    #     print(False)
    #     hier_flag = False
    # hier_flag = check_layer_callback_k(k1, k2)
    # print(tmp_time_list)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    if plot_flag:
        plt.plot(range(1, len(ori_tmp_time_list) + 1), ori_tmp_time_list, label='ori_ALL')
        # plt.plot(range(1, len(ori_tmp_time_list) + 1), ori_tmp_time_list, label='ALL')
        plt.plot(range(k1, k2), ori_tmp_time_list[k1 - 1:k2 - 1], label='Adaption', marker='x', color='r')
        # plt.plot(final_time_1)
        plt.hlines(final_time_1, color='k', xmin=1, xmax=len(ori_tmp_time_list) + 1, label='Origin')
        plt.legend()
        plt.show()
    return ori_tmp_time_list, [k1, k2]


def get_check_result(data_num_list, test_num=20, dist_flag='t', dist_para=[10]):
    """

    :param data_num_list: 客户端数量
    :param test_num: 每个客户端数量实验的次数
    :return:
    """
    res_list = []
    for i in data_num_list:
        tmp_list = []
        for j in range(test_num):
            tmp_hier_flag = test_check_hier(data_num=i, dist_flag=dist_flag, dis_para=dist_para)
            tmp_list.append(tmp_hier_flag)
        res_list.append(tmp_list)
    flag_res_list = np.sum(res_list, axis=1) / test_num
    plt.plot(data_num_list, flag_res_list, label='ALL', marker='.')
    plt.show()
    return flag_res_list


# def get_two_layer_result(client_num, dist_flag='t', dist_para=10):
#     """
#
#     :param data_num_list: 客户端数量
#     :param test_num: 每个客户端数量实验的次数
#     :return:
#     """
#     res_list = []
#     tmp_list = []
#     tmp_time_list, k = test_check_two_layer_adaptive(data_num=client_num, dist_flag=dist_flag, dis_para=dist_para)
#     tmp_list.append(tmp_hier_flag)
#     res_list.append(tmp_list)
#     flag_res_list = np.sum(res_list, axis=1)/test_num
#     plt.plot(data_num_list, flag_res_list, label='ALL', marker='.')
#     plt.show()
#     return flag_res_list

def get_agg_time_con(param_size, client_num):
    param_size_list = [client_num]
    for i in param_size:
        param_size_list.append(i)
    tmp_para = np.random.random(param_size_list)
    time_start = time.time()
    sum_para = np.mean(tmp_para, axis=0)
    time_end = time.time()
    print('time_end - time_start: {}'.format(time_end - time_start))


def obtain_distributions_raw_data(client_num, dis_flag, dis_para):
    size = (client_num, 2)
    if dis_flag == 'gamma':
        raw_data = np.random.standard_gamma(shape=size)
        # raw_data = np.abs(np.random.randn(data_num, 2))
        raw_data[:, 1] = 20 * raw_data[:, 1]
    elif dis_flag == 'norm':
        # 通过以上参数解释可知：np.random.normal(loc=0, scale=1, size)就表示标准正太分布（μ=0, σ=1）。
        raw_data = np.random.normal(loc=dis_para[0], scale=dis_para[1], size=size)
    elif dis_flag == 't':
        raw_data = np.random.standard_t(dis_para, size=size)
    else:
        raw_data = np.random.rand(client_num, 2)
        # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = np.abs(raw_data)
    return raw_data


def obtain_norm_distributions_raw_data(client_num, dis_flag, dis_para):
    size = (client_num, 2)
    if dis_flag == 'norm':
        # 通过以上参数解释可知：np.random.normal(loc=0, scale=1, size)就表示标准正太分布（μ=0, σ=1）。
        raw_data = np.random.normal(loc=dis_para[0], scale=dis_para[1], size=size)

    else:
        raw_data = np.random.rand(client_num, 2)
        # raw_data = np.abs(np.random.randn(data_num, 2))
    raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = np.abs(raw_data)
    return raw_data


def obtain_k_time(args, start=2, stop=50, step=1):
    # 验证分层条件的有效性
    # 使用正太分布
    dist_flag = 'norm'  # 'gamma', 'norm', 't'
    data_num_list = np.arange(start=start, stop=stop, step=step)

    dist_para = [[10, 0.5], [10, 1], [20, 0.5], [20, 1]]
    res_dict = {'client_number': data_num_list.tolist()}
    for tmp_dist_para in dist_para:
        tmp_flag_res_list = get_check_result(data_num_list=data_num_list, dist_flag=dist_flag, dist_para=tmp_dist_para)
        res_dict[str(tmp_dist_para)] = tmp_flag_res_list.tolist()

    res_df = pd.DataFrame(res_dict)
    # res_df.plot()
    # plt.show()
    save_path = os.path.join(main_res_path, '{}.xlsx'.format(args.eval_item))
    res_df.to_csv(save_path, index=False)


def obtain_two_layer_effective_time(args, client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]]):
    # 验证分层条件的有效性
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
    # 使用正太分布
    dist_flag = 'norm'  # 'gamma', 'norm', 't'
    # save_path = os.path.join(main_res_path, '{}.xlsx')
    save_path = os.path.join(main_res_path, '{}.xlsx'.format(args.eval_item))
    writer = pd.ExcelWriter(save_path)
    for client_num in client_num_list:
        print('current client_num is {}'.format(client_num))
        res_dict = {}
        for tmp_dist_para in dist_para:
            print('current client_num is {}, current tmp_dist_para is {}'.format(client_num, tmp_dist_para))
            tmp_time_list, k = test_check_two_layer_adaptive(data_num=client_num, dist_flag=dist_flag,
                                                             dis_para=tmp_dist_para, plot_flag=False)
            # tmp_time_list.append(k, )  方法向列表的尾部添加一个新的元素。
            tmp_time_list.extend(k)  # 方法只接受一个列表作为参数，并将该参数的每个元素都添加到原有的列表中。

            res_dict[str(tmp_dist_para)] = tmp_time_list
        res_df = pd.DataFrame(res_dict)
        res_df.to_excel(excel_writer=writer, sheet_name=str(client_num), index=False)
    writer.save()


def obtain_multiple_layered_effective_time(client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]], num=2):

    # 验证分层条件的有效性
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
    # 使用正太分布
    dist_flag = 'norm'  # 'gamma', 'norm', 't'
    # dist_para = [[10, 0.5], [10, 1], [20, 0.5], [20, 1]]
    # res_dict = {'client_number': data_num_list}
    save_path = os.path.join(main_res_path, 'multiple_layered_adp_{}.xlsx'.format(num))
    writer = pd.ExcelWriter(save_path)
    sheet_name_list = []
    for tmp_dist_para in dist_para:

        for client_num in client_num_list:
            cloud_time_list = []
            two_layer_list = []
            layered_list = []
            print('current client_num is {}, current tmp_dist_para is {}'.format(client_num, tmp_dist_para))
            # two layer 时间消耗
            for tmp_num in range(num):
                cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                               dist_flag=dist_flag,
                                                                                               dis_para=tmp_dist_para)
                # tmp_time_list.append(k, )  方法向列表的尾部添加一个新的元素。
                cloud_time_list.append(cloud_time)
                two_layer_list.append(two_layer_tmp_time)
                layered_list.append(layered_time_con)

            res_dict = {'cloud_time': cloud_time_list, 'two_layer': two_layer_list,
                        'layered': layered_list}
            res_df = pd.DataFrame(res_dict)

            sheet_name = str(tmp_dist_para[0]) + '_' + str(tmp_dist_para[1]) + '_' + str(client_num)
            res_df.to_excel(excel_writer=writer, sheet_name=sheet_name, index=False)
            print('sheet_name={}'.format(sheet_name))
            print(res_df)
            sheet_name_list.append(sheet_name)
    writer.save()
    writer.close()
    print('over')
    # extract_data(save_path, sheet_name_list)


def obtain_layered_effective_time(client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]]):
    # 验证分层条件的有效性
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
    # 使用正太分布
    dist_flag = 'norm'  # 'gamma', 'norm', 't'
    # dist_para = [[10, 0.5], [10, 1], [20, 0.5], [20, 1]]
    # res_dict = {'client_number': data_num_list}
    save_path = os.path.join(main_res_path, 'layered_adp.xlsx')
    writer = pd.ExcelWriter(save_path)
    sheet_name_list = []
    for tmp_dist_para in dist_para:
        res_dict = {}
        cloud_time_list = []
        two_layer_list = []
        layered_list = []
        for client_num in client_num_list:
            print('current client_num is {}, current tmp_dist_para is {}'.format(client_num, tmp_dist_para))
            # two layer 时间消耗
            cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                           dist_flag=dist_flag,
                                                                                           dis_para=tmp_dist_para)
            # tmp_time_list.append(k, )  方法向列表的尾部添加一个新的元素。
            cloud_time_list.append(cloud_time)
            two_layer_list.append(two_layer_tmp_time)
            layered_list.append(layered_time_con)

            res_dict = {'client_num': client_num_list, 'cloud_time': cloud_time_list, 'two_layer': two_layer_list,
                        'layered': layered_list}
        res_df = pd.DataFrame(res_dict)

        sheet_name = str(tmp_dist_para[0]) + '_' + str(tmp_dist_para[1])
        res_df.to_excel(excel_writer=writer, sheet_name=sheet_name, index=False)
        print('sheet_name={}'.format(sheet_name))
        print(res_df)
        sheet_name_list.append(sheet_name)
    writer.save()
    writer.close()
    print('over')
    extract_data(save_path, sheet_name_list)

def extract_data(file_name: str, sheet_name_list):
    data_list = []
    index_list = []      # 提取所有索引保存在list中
    # 提取所有数据集保存在list中
    for sheet_name in sheet_name_list:
        # res_df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl')
        res_df = pd.read_excel(file_name, sheet_name=sheet_name)
        tmp_index = res_df.iloc[: , 0].values.tolist()
        index_list.append(tmp_index)
        data_list.append(res_df)
    save_file_name = file_name.replace('.xlsx', '_handle.xlsx')
    writer = pd.ExcelWriter(save_file_name)
    index = index_list[0]
    column_list = data_list[0].columns.tolist()
    for tmp_index in index:
        save_sheet_name = str(tmp_index)
        tmp_res_list = {}
        for tmp_resdf, sheet_name in zip(data_list, sheet_name_list):
            tmp_data = tmp_resdf[tmp_resdf.iloc[:, 0] == tmp_index].iloc[0, 1:].values.tolist()
            # tmp_data = tmp_resdf[tmp_resdf[0] == tmp_index].values.tolist()
            # tmp_data = list(map(list, zip(*tmp_data)))
            # tmp_data = tmp_data.reshape(tmp_data.shape[0],1)
            # tmp_res_list.append({str(sheet_name): tmp_data})
            tmp_res_list[str(sheet_name)] = tmp_data
        handle_df = pd.DataFrame(tmp_res_list, columns=sheet_name_list)
        handle_df.insert(loc=0, column='method', value=column_list[1:] )
        handle_df.to_excel(excel_writer=writer, index=False, sheet_name=save_sheet_name)
    writer.save()
    writer.close()

def avg_data(file_name: str, sheet_name_list):
    # 在原来的df中添加平均值
    save_file_name = file_name.replace('.xlsx', '_avg.xlsx')
    writer = pd.ExcelWriter(save_file_name)

    for sheet_name in sheet_name_list:
        # res_df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl')
        res_df = pd.read_excel(file_name, sheet_name=sheet_name)
        avg_tmp = res_df.mean()
        res_df = res_df.append(avg_tmp, ignore_index=True)
        res_df.to_excel(excel_writer=writer, index=False, sheet_name=sheet_name)
    writer.save()
    writer.close()

def avg_extract_data(file_name, client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]]):
    save_file_name = file_name.replace('.xlsx', '_handle_avg.xlsx')
    writer = pd.ExcelWriter(save_file_name)
    for client_num in client_num_list:
        tmp_data_list = {}
        dis_list = []
        for tmp_dist_para in dist_para:
            tmp_name = str(tmp_dist_para[0]) + '_' + str(tmp_dist_para[1])
            sheet_name = tmp_name + '_' + str(client_num)
            dis_list.append(tmp_name)

            res_df = pd.read_excel(file_name, sheet_name=sheet_name)
            columns_list = res_df.columns
            avg_tmp = res_df.mean().values.tolist()
            # res_df = res_df.append(avg_tmp, ignore_index=True)
            tmp_data_list[tmp_name] = avg_tmp
        new_res_df = pd.DataFrame(tmp_data_list, columns=dis_list)
        new_res_df.insert(loc=0, column='method', value= columns_list)

        new_res_df.to_excel(excel_writer=writer, index=False, sheet_name=str(client_num))
    writer.save()
    writer.close()


if __name__ == '__main__':
    # stop_condition: python layered_aggregation.py -e stop_condition
    # two_layer: python layered_aggregation.py -e two_layer
    # layered: python layered_aggregation.py -e layered
    parser = argparse.ArgumentParser(description='Evaluate the time consumption of the proposed decentralized layered mode aggregation framework')
    parser.add_argument("-r", "--res", default="./res", type=str)
    parser.add_argument("-e", "--eval_item", choices=['stop_condition', 'two_layer', 'layered'], default="stop_condition",
                        type=str, help='stop_condition: evaluating the rate of further dividing a cluster with different numbers of nodes.'
                                       'two_layer: the time consumption of two-layer model aggregation framework'
                                       'layered: the time consumption comparisons between the two-layer model aggregation framework and the decentralized layered model aggregation framework')
    parser.add_argument("-s1", "--start_n", default=2, type=int)
    parser.add_argument("-s2", "--step_n", default=1, type=int)
    parser.add_argument("-s3", "--stop_n", default=100, type=int)
    parser.add_argument("-cnl", "--client_num_list", default=[50, 100, 1000, 10000], type=list, help='the number of nodes')
    parser.add_argument("-dp", "--dist_para", default=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]], type=list, help='different combinations of $\mu$ and $\sigma$')
    args = parser.parse_args()
    print(args)
    main_res_path = args.res
    if not os.path.exists(main_res_path):
        os.makedirs(main_res_path)
    if args.eval_item == 'stop_condition':
        obtain_k_time(args=args, start=args.start_n, stop=args.stop_n, step =args.step_n)

    elif args.eval_item == 'two_layer':
        obtain_two_layer_effective_time(args, args.client_num_list, args.dist_para)

    elif args.eval_item == 'layered':
        dist_para = args.dist_para
        num = 20  # Number of experiment repetitions
        client_num_list = args.client_num_list
        obtain_multiple_layered_effective_time(client_num_list=client_num_list, dist_para=dist_para, num=num)
        save_path = os.path.join(main_res_path, 'multiple_layered_adp_{}.xlsx'.format(num))
        avg_extract_data(file_name=save_path, client_num_list=client_num_list, dist_para=dist_para)

    else:
        raise NotImplementedError

