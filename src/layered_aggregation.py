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

    time_con = np.sum(raw_data, axis=1)
    max_time_con = np.max(time_con)

    t_u_0 = np.min(raw_data[:, 0])
    max_time_con2 = t_u_0 + np.sum(raw_data[:, 1])

    final_time = np.max([max_time_con, max_time_con2])

    return final_time


def obtain_precision_raw_time(raw_data):

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
                current_time = ascend_raw_data[i, 0]
            current_time += ascend_raw_data[i, 1]  # 启动上传工作、
        else:
            finish_update_node = i
            break

    time_con = np.sum(raw_data[finish_update_node:, 1])
    current_time += time_con
    return current_time


def obtain_k_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
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
    :param k:
    :param ascend_raw_data:
    :return:
    """
    n_c = ascend_raw_data.shape[0]
    a = int(np.ceil((n_c - k) / k))
    index_list = np.arange(ascend_raw_data.shape[0])
    oridinary_index_list = index_list[k:]
    #  data partition：https://blog.csdn.net/mingyuli/article/details/81227629
    np.random.shuffle(oridinary_index_list)
    split_slice_list = [i * a for i in range(1, k)]
    split_array = np.split(oridinary_index_list, split_slice_list)
    for i in range(k):
        split_array[i] = np.append(split_array[i], i)
    time_list = []
    for i in range(k):
        cur_split_array = split_array[i]
        cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
        tmp_raw_data = ascend_raw_data[cur_flag_list]
        cur_time_con = obtain_precision_raw_time(tmp_raw_data)
        time_list.append(cur_time_con)
        print('k: {}, time_con: {}'.format(k, cur_time_con))
    new_raw_data = np.array([time_list, ascend_raw_data[:k, 1].tolist()]).T
    second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


def branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=False):
    """
    :param ascend_raw_data:
    :param cur_layer_flag_list:
    :param leader_layer_flag:
    :return:
    """
    cur_time_con_list = []
    leader_node_raw_data = []

    for tmp_raw_flag in cur_layer_flag_list:

        tmp_raw_data = ascend_raw_data[tmp_raw_flag]
        leader_node_raw_data.append(tmp_raw_data[0, :].tolist())

        tmp_hier_flag, tmp_cur_layer_flag_list, tmp_cur_time_con = layer_allocation(tmp_raw_data)

        tmp_n_c = tmp_raw_data.shape[0]
        print('hier_flag: {}, nc: {}, k: {}'.format(tmp_hier_flag, tmp_n_c, len(tmp_cur_layer_flag_list)))

        if tmp_hier_flag:
            tmp_cur_time_con = branch_layer_time_con(tmp_raw_data, tmp_cur_layer_flag_list)
        cur_time_con_list.append(tmp_cur_time_con)
    if leader_layer_flag:
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
    else:
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


def layer_allocation(ascend_raw_data):
    """
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
        #  datapartition：https://blog.csdn.net/mingyuli/article/details/81227629
        np.random.shuffle(oridinary_index_list)
        split_slice_list = [i * a for i in range(1, k)]
        split_array = np.split(oridinary_index_list, split_slice_list)
        for i in range(k):
            split_array[i] = np.append(split_array[i], i)
        for i in range(k):
            cur_split_array = split_array[i]
            cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
            cur_layer_flag_list.append(cur_flag_list)

    else:
        cur_time_con = obtain_precision_raw_time(ascend_raw_data)

    return hier_flag, cur_layer_flag_list, cur_time_con


def layered_adapt_time_con(ascend_raw_data):
    """
    :param k:
    :param ascend_raw_data:
    :return:
    """
    hier_flag, cur_layer_flag_list, cur_time_con = layer_allocation(ascend_raw_data)
    if hier_flag:
        final_time = branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=True)
    else:
        final_time = cur_time_con
    return final_time


def obtain_k_precision_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    tmp_time_list = []
    E_t_d = np.mean(raw_data[:, 1])
    max_t_u = np.max(raw_data[:, 0])
    n_c = raw_data.shape[0]
    for k in range(1, ascend_raw_data.shape[0]):
        temp_time_con = two_adapt_time_con(k, ascend_raw_data)
        tmp_time_list.append(temp_time_con)
    optimal_k = np.argmin(tmp_time_list) + 1
    print('optimal_k is {}'.format(optimal_k))
    print('optimal time consumption is {}'.format(tmp_time_list[optimal_k - 1]))
    return tmp_time_list


def obtain_k_single_precision_syn_method(raw_data, k):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    temp_time_con = two_adapt_time_con(k, ascend_raw_data)
    return temp_time_con


def obtain_layered_precision_syn_method(raw_data):
    """
    :param raw_data:
    :return:
    """

    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    temp_time_con = layered_adapt_time_con(ascend_raw_data)
    return temp_time_con

def obtain_adapt_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    k_ = np.argmax(ascend_raw_data[:, 1])
    n_c = raw_data.shape[0]
    if n_c == 1:  #
        return 0, 0
    E_t_d = np.mean(raw_data[:, 1])
    max_t_d = np.max(ascend_raw_data[:, 1])
    min_t_d = np.min(ascend_raw_data[:, 1])
    max_t_u = np.max(ascend_raw_data[:, 0])
    min_t_u = np.min(ascend_raw_data[:, 0])
    # when k>K_
    k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u)), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # if k1 < k_:
    #     k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u + (k_-1)*(max_t_d-min_t_d))), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + (1 - k_ / n_c) * max_t_u)), np.sqrt(n_c * E_t_d / max_t_d)

    print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))

    if np.ceil(k1) < k_:
        # temp_k1 = n_c * n_c * E_t_d / (k_ * max_t_d + (k_ * max_t_u - n_c * min_t_u) + (k_ - 1) * (
        #             k_ * max_t_d - n_c * min_t_d))
        temp_k1 = n_c * E_t_d / (max_t_d + (max_t_u - n_c * min_t_u) + (k_ - 1) * (
                max_t_d - n_c * min_t_d))
        if temp_k1 > 0:
            k1 = np.sqrt(temp_k1)
        else:
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
    if np.isnan(k1) or k1 > k2:
        print('error')
        print(k1)
    x = sp.Symbol('x', positive=True)
    # f = 2*n_c/(x**3) + max_t_d - min_t_d
    f_test = max_t_u - min_t_u - n_c / (1 ** 2) * E_t_d + max_t_d + (1 - 1) * (max_t_d - min_t_d)
    print('f_test: {}'.format(f_test))
    f = max_t_u - min_t_u - n_c / (x ** 2) * E_t_d + max_t_d + (x - 1) * (max_t_d - min_t_d)
    # sp.solve(f)
    x_test = sp.nsolve(f, x, k1, prec=3, verify=False)
    # if x_test > k1:
    if not math.isnan(x_test):
        k1 = x_test
    print('x_test: {}'.format(x_test))
    f_test = max_t_u - min_t_u - n_c / (x_test ** 2) * E_t_d + max_t_d + (x_test - 1) * (max_t_d - min_t_d)
    print('f_test2: {}'.format(f_test))
    # reciprocal of k
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

    return k1, k2


def check_layer_callback(k1, k2):
    """

    :param k1: upper bound
    :param k2: lower bound
    :return:
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
    :param k1: upper bound
    :param k2: lower bound
    :return:
    """
    if np.floor(k2) >= 2 and np.ceil(k2) - np.ceil(k1) >= 1:
        print(True)
        hier_flag = True
    else:
        print(False)
        hier_flag = False

    return hier_flag


def test_check_hier(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_raw_time(raw_data)
    # print('Single aggregation node time is {}'.format(final_time_1))
    # tmp_time_list = obtain_k_syn_method(raw_data)
    k1, k2 = obtain_adapt_syn_method(raw_data)

    hier_flag = check_layer_callback_k(k1, k2)

    return hier_flag


def test_check_two_layer_adaptive_precision(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """
    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
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

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_precision_raw_time(raw_data)
    final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    k1, k2 = obtain_adapt_syn_method(raw_data)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准，但比较快
    two_layer_tmp_time = obtain_k_single_precision_syn_method(raw_data, k2)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    layered_time_con = obtain_layered_precision_syn_method(raw_data)

    return final_time_1, two_layer_tmp_time, layered_time_con


def test_check_two_layer_adaptive(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)
    ori_tmp_time_list = obtain_k_syn_method(raw_data)
    k1, k2 = obtain_adapt_syn_method(raw_data)

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

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """

    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准, 但比较快
    k1, k2 = obtain_adapt_syn_method(raw_data)

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

    :param data_num: the number of clients
    :param plot_flag:
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
        raw_data = np.random.normal(loc=dis_para[0], scale=dis_para[1], size=size)

    else:
        raw_data = np.random.rand(client_num, 2)
        # raw_data = np.abs(np.random.randn(data_num, 2))
    raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = np.abs(raw_data)
    return raw_data


def obtain_k_time(args, start=2, stop=50, step=1):
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
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            tmp_time_list.extend(k)

            res_dict[str(tmp_dist_para)] = tmp_time_list
        res_df = pd.DataFrame(res_dict)
        res_df.to_excel(excel_writer=writer, sheet_name=str(client_num), index=False)
    writer.save()


def obtain_multiple_layered_effective_time(client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]], num=2):

    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            # two layer time consumption
            for tmp_num in range(num):
                cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                               dist_flag=dist_flag,
                                                                                               dis_para=tmp_dist_para)
                # tmp_time_list.append(k, )
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
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            # two layer time consumption
            cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                           dist_flag=dist_flag,
                                                                                           dis_para=tmp_dist_para)
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
    index_list = []
    for sheet_name in sheet_name_list:
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

    time_con = np.sum(raw_data, axis=1)
    max_time_con = np.max(time_con)

    t_u_0 = np.min(raw_data[:, 0])
    max_time_con2 = t_u_0 + np.sum(raw_data[:, 1])

    final_time = np.max([max_time_con, max_time_con2])

    return final_time


def obtain_precision_raw_time(raw_data):

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
                current_time = ascend_raw_data[i, 0]
            current_time += ascend_raw_data[i, 1]  # 启动上传工作、
        else:
            finish_update_node = i
            break

    time_con = np.sum(raw_data[finish_update_node:, 1])
    current_time += time_con
    return current_time


def obtain_k_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
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
    :param k:
    :param ascend_raw_data:
    :return:
    """
    n_c = ascend_raw_data.shape[0]
    a = int(np.ceil((n_c - k) / k))
    index_list = np.arange(ascend_raw_data.shape[0])
    oridinary_index_list = index_list[k:]
    #  data partition：https://blog.csdn.net/mingyuli/article/details/81227629
    np.random.shuffle(oridinary_index_list)
    split_slice_list = [i * a for i in range(1, k)]
    split_array = np.split(oridinary_index_list, split_slice_list)
    for i in range(k):
        split_array[i] = np.append(split_array[i], i)
    time_list = []
    for i in range(k):
        cur_split_array = split_array[i]
        cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
        tmp_raw_data = ascend_raw_data[cur_flag_list]
        cur_time_con = obtain_precision_raw_time(tmp_raw_data)
        time_list.append(cur_time_con)
        print('k: {}, time_con: {}'.format(k, cur_time_con))
    new_raw_data = np.array([time_list, ascend_raw_data[:k, 1].tolist()]).T
    second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


def branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=False):
    """
    :param ascend_raw_data:
    :param cur_layer_flag_list:
    :param leader_layer_flag:
    :return:
    """
    cur_time_con_list = []
    leader_node_raw_data = []

    for tmp_raw_flag in cur_layer_flag_list:

        tmp_raw_data = ascend_raw_data[tmp_raw_flag]
        leader_node_raw_data.append(tmp_raw_data[0, :].tolist())

        tmp_hier_flag, tmp_cur_layer_flag_list, tmp_cur_time_con = layer_allocation(tmp_raw_data)

        tmp_n_c = tmp_raw_data.shape[0]
        print('hier_flag: {}, nc: {}, k: {}'.format(tmp_hier_flag, tmp_n_c, len(tmp_cur_layer_flag_list)))

        if tmp_hier_flag:
            tmp_cur_time_con = branch_layer_time_con(tmp_raw_data, tmp_cur_layer_flag_list)
        cur_time_con_list.append(tmp_cur_time_con)
    if leader_layer_flag:
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
    else:
        new_raw_data = np.array([cur_time_con_list, np.array(leader_node_raw_data)[:, 1].tolist()]).T
        second_stage_time = obtain_precision_raw_time(new_raw_data)
    final_time = second_stage_time
    return final_time


def layer_allocation(ascend_raw_data):
    """
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
        #  datapartition：https://blog.csdn.net/mingyuli/article/details/81227629
        np.random.shuffle(oridinary_index_list)
        split_slice_list = [i * a for i in range(1, k)]
        split_array = np.split(oridinary_index_list, split_slice_list)
        for i in range(k):
            split_array[i] = np.append(split_array[i], i)
        for i in range(k):
            cur_split_array = split_array[i]
            cur_flag_list = [True if i in cur_split_array else False for i in range(n_c)]
            cur_layer_flag_list.append(cur_flag_list)

    else:
        cur_time_con = obtain_precision_raw_time(ascend_raw_data)

    return hier_flag, cur_layer_flag_list, cur_time_con


def layered_adapt_time_con(ascend_raw_data):
    """
    :param k:
    :param ascend_raw_data:
    :return:
    """
    hier_flag, cur_layer_flag_list, cur_time_con = layer_allocation(ascend_raw_data)
    if hier_flag:
        final_time = branch_layer_time_con(ascend_raw_data, cur_layer_flag_list, leader_layer_flag=True)
    else:
        final_time = cur_time_con
    return final_time


def obtain_k_precision_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    tmp_time_list = []
    E_t_d = np.mean(raw_data[:, 1])
    max_t_u = np.max(raw_data[:, 0])
    n_c = raw_data.shape[0]
    for k in range(1, ascend_raw_data.shape[0]):
        temp_time_con = two_adapt_time_con(k, ascend_raw_data)
        tmp_time_list.append(temp_time_con)
    optimal_k = np.argmin(tmp_time_list) + 1
    print('optimal_k is {}'.format(optimal_k))
    print('optimal time consumption is {}'.format(tmp_time_list[optimal_k - 1]))
    return tmp_time_list


def obtain_k_single_precision_syn_method(raw_data, k):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    temp_time_con = two_adapt_time_con(k, ascend_raw_data)
    return temp_time_con


def obtain_layered_precision_syn_method(raw_data):
    """
    :param raw_data:
    :return:
    """

    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    temp_time_con = layered_adapt_time_con(ascend_raw_data)
    return temp_time_con

def obtain_adapt_syn_method(raw_data):
    raw_data: np.array
    raw_data_df = pd.DataFrame(raw_data)
    ascend_raw_data = raw_data_df.sort_values(by=0).values
    k_ = np.argmax(ascend_raw_data[:, 1])
    n_c = raw_data.shape[0]
    if n_c == 1:  #
        return 0, 0
    E_t_d = np.mean(raw_data[:, 1])
    max_t_d = np.max(ascend_raw_data[:, 1])
    min_t_d = np.min(ascend_raw_data[:, 1])
    max_t_u = np.max(ascend_raw_data[:, 0])
    min_t_u = np.min(ascend_raw_data[:, 0])
    # when k>K_
    k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u)), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # if k1 < k_:
    #     k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + max_t_u - min_t_u + (k_-1)*(max_t_d-min_t_d))), np.sqrt(n_c * E_t_d / max_t_d)
    # print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))
    # k1, k2 = np.sqrt(n_c * E_t_d / (max_t_d + (1 - k_ / n_c) * max_t_u)), np.sqrt(n_c * E_t_d / max_t_d)

    print('k_, k1, k2: {}, {}, {}'.format(k_, k1, k2))

    if np.ceil(k1) < k_:
        # temp_k1 = n_c * n_c * E_t_d / (k_ * max_t_d + (k_ * max_t_u - n_c * min_t_u) + (k_ - 1) * (
        #             k_ * max_t_d - n_c * min_t_d))
        temp_k1 = n_c * E_t_d / (max_t_d + (max_t_u - n_c * min_t_u) + (k_ - 1) * (
                max_t_d - n_c * min_t_d))
        if temp_k1 > 0:
            k1 = np.sqrt(temp_k1)
        else:
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
    if np.isnan(k1) or k1 > k2:
        print('error')
        print(k1)
    x = sp.Symbol('x', positive=True)
    # f = 2*n_c/(x**3) + max_t_d - min_t_d
    f_test = max_t_u - min_t_u - n_c / (1 ** 2) * E_t_d + max_t_d + (1 - 1) * (max_t_d - min_t_d)
    print('f_test: {}'.format(f_test))
    f = max_t_u - min_t_u - n_c / (x ** 2) * E_t_d + max_t_d + (x - 1) * (max_t_d - min_t_d)
    # sp.solve(f)
    x_test = sp.nsolve(f, x, k1, prec=3, verify=False)
    # if x_test > k1:
    if not math.isnan(x_test):
        k1 = x_test
    print('x_test: {}'.format(x_test))
    f_test = max_t_u - min_t_u - n_c / (x_test ** 2) * E_t_d + max_t_d + (x_test - 1) * (max_t_d - min_t_d)
    print('f_test2: {}'.format(f_test))
    # reciprocal of k
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

    return k1, k2


def check_layer_callback(k1, k2):
    """

    :param k1: upper bound
    :param k2: lower bound
    :return:
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
    :param k1: upper bound
    :param k2: lower bound
    :return:
    """
    if np.floor(k2) >= 2 and np.ceil(k2) - np.ceil(k1) >= 1:
        print(True)
        hier_flag = True
    else:
        print(False)
        hier_flag = False

    return hier_flag


def test_check_hier(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_raw_time(raw_data)
    # print('Single aggregation node time is {}'.format(final_time_1))
    # tmp_time_list = obtain_k_syn_method(raw_data)
    k1, k2 = obtain_adapt_syn_method(raw_data)

    hier_flag = check_layer_callback_k(k1, k2)

    return hier_flag


def test_check_two_layer_adaptive_precision(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """
    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
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

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    # final_time_1 = obtain_precision_raw_time(raw_data)
    final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    k1, k2 = obtain_adapt_syn_method(raw_data)
    k1, k2 = int(np.ceil(k1)), int(np.ceil(k2))
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准，但比较快
    two_layer_tmp_time = obtain_k_single_precision_syn_method(raw_data, k2)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    layered_time_con = obtain_layered_precision_syn_method(raw_data)

    return final_time_1, two_layer_tmp_time, layered_time_con


def test_check_two_layer_adaptive(data_num=1000, plot_flag=False, dist_flag=None, dis_para=None):
    """

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """
    # data_num = 200000
    # data_num = 1000
    # data_num = 100
    # np.random.seed(914)
    # raw_data = np.random.rand(data_num, 2)
    # # raw_data = np.abs(np.random.randn(data_num, 2))
    # raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    # ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)
    ori_tmp_time_list = obtain_k_syn_method(raw_data)
    k1, k2 = obtain_adapt_syn_method(raw_data)

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

    :param data_num: the number of clients
    :param plot_flag:
    :return:
    """

    raw_data = obtain_distributions_raw_data(client_num=data_num, dis_flag=dist_flag, dis_para=dis_para)
    final_time_1 = obtain_precision_raw_time(raw_data)
    # final_time_1 = obtain_raw_time(raw_data)
    print('Single aggregation node time is {}'.format(final_time_1))
    ori_tmp_time_list = obtain_k_precision_syn_method(raw_data)  # 这个会花费很长时间的哦，而效果跟上面那个是相似的
    # ori_tmp_time_list = obtain_k_syn_method(raw_data)  # 不够精准, 但比较快
    k1, k2 = obtain_adapt_syn_method(raw_data)

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

    :param data_num: the number of clients
    :param plot_flag:
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
        raw_data = np.random.normal(loc=dis_para[0], scale=dis_para[1], size=size)

    else:
        raw_data = np.random.rand(client_num, 2)
        # raw_data = np.abs(np.random.randn(data_num, 2))
    raw_data[:, 1] = 20 * raw_data[:, 1]
    raw_data = np.abs(raw_data)
    return raw_data


def obtain_k_time(args, start=2, stop=50, step=1):
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
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            tmp_time_list.extend(k)

            res_dict[str(tmp_dist_para)] = tmp_time_list
        res_df = pd.DataFrame(res_dict)
        res_df.to_excel(excel_writer=writer, sheet_name=str(client_num), index=False)
    writer.save()


def obtain_multiple_layered_effective_time(client_num_list=[50, 100], dist_para=[[10, 0.5], [10, 1], [20, 0.5], [20, 1]], num=2):

    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            # two layer time consumption
            for tmp_num in range(num):
                cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                               dist_flag=dist_flag,
                                                                                               dis_para=tmp_dist_para)
                # tmp_time_list.append(k, )
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
    # dist_flag = 't' # 'gamma', 'norm', 't'
    # dist_para = 50
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
            # two layer time consumption
            cloud_time, two_layer_tmp_time, layered_time_con = test_check_layered_adaptive(data_num=client_num,
                                                                                           dist_flag=dist_flag,
                                                                                           dis_para=tmp_dist_para)
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
    index_list = []
    for sheet_name in sheet_name_list:
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

