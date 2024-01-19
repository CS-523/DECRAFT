import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from torch.optim.lr_scheduler import MultiStepLR


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):
        """
           Initialize the posencoding component
           :param dim: the encoding dimension
           :param device: where to train model
           :param base: the encoding base
           :param bias: the encoding bias
       """
        super(PosEncoding, self).__init__()

        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransEmbedding(nn.Module):

    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82, cat_features=None):
        """
        Initialize the attribute embedding and feature learning compoent

        :param df: the feature
                :param device: where to train model
                :param dropout: the dropout rate
                :param in_feat: the shape of input feature in dimension 1
                :param cat_feature: category features including [card、transaction、merchant ]
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        # time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        )) + 1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, df):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        # print(self.emb_dict)
        # print(df['trans_md'])
        support = {col: self.emb_dict[col](
            df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        # self.time_emb = self.time_pe(torch.sin(torch.tensor(df['time_span'])/86400*torch.pi))
        # support['time_span'] = self.time_emb
        # support['labels'] = self.label_table(df['labels'])
        return support

    def forward(self, df):
        support = self.forward_emb(df)
        output = 0
        for i, k in enumerate(support.keys()):
            # if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]
        return output


class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention 
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature 
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        # self.feat_dropout = nn.Dropout(p=feat_drop)
        # self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention
        """
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        a = graph.edata['a']
        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats ** 0.5)
        sa = graph.edata['sa']
        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))
        agg_u = graph.dstdata['agg_u']
        # output results to the destination nodes: concat
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats * self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class DenTargetTransformerConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function
        """

        super(DenTargetTransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        # self.lin_query = nn.Linear(
        #     self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        # self.lin_value = nn.Linear(
        #     self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        # self.feat_dropout = nn.Dropout(p=feat_drop)
        # self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, q_src, v_src, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param q_src:
            :param v_src:
            :param get_attention: whether to get attention
        """
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # q_src = self.lin_query(
        #     h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        # v_src = self.lin_value(
        #     h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        # graph.srcdata.update({'ft': q_src})
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        a = graph.edata['a']
        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats ** 0.5)
        sa = graph.edata['sa']
        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))
        agg_u = graph.dstdata['agg_u']
        # output results to the destination nodes: concat
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats * self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class DenNeighborTransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 ):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function
        """

        super(DenNeighborTransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

    def forward(self, graph, feat):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
        """
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
        else:
            h_src = feat
            # h_dst = h_src[:graph.number_of_dst_nodes()]
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        # k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)

        return q_src, v_src


class DecentralizedFeatureExtration(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers (the numer of the layered neighbors)
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
        :param device: where to train model
        """

        super(DecentralizedFeatureExtration, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

    def forward(self, features, n2v_feat=None):
        """
        :param features: train features  (|input|, feta_dim)
        :param n2v_feat: whether to use n2v features
        """
        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h

        h = self.layers[1](h)
        h = self.layers[3](h)
        return h


class ModelBuildingDecentralize():
    def __init__(self):
        pass

    def build_node_optimizer(self, model_list, lr, args):
        optimizer_list = {}
        lr_scheduler_list = {}
        gamma = 0.3
        for tmp_model in model_list:
            model = model_list[tmp_model]
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=args['wd'])
            optimizer_list[tmp_model] = optimizer
            lr_scheduler = MultiStepLR(optimizer=optimizer_list[tmp_model], milestones=[
                4000, 12000], gamma=gamma)
            lr_scheduler_list[tmp_model] = lr_scheduler
        return optimizer_list, lr_scheduler_list

    def build_node_model_train(self, model_list):
        for tmp_model in model_list:
            model = model_list[tmp_model]
            model.train()

    def build_node_model_eval(self, model_list):
        for tmp_model in model_list:
            model = model_list[tmp_model]
            model.eval()

    def model_farward(self, model_list, blocks, lpa_labels):
        """
        :param model_list: model num model，
        :param blocks:
        :param batch_inputs:
        :param lpa_labels:
        :param batch_work_inputs:
        :return:
        """
        embedding_list = []
        for tmp_model_num in range(len(model_list) - 1):
            graph = blocks[tmp_model_num]
            tmp_num = int(graph.srcdata['feat'].shape[0] - graph.dstdata['feat'].shape[0])
            if tmp_num == 0:
                # h = []
                # print('tmp_num is zero!!')
                raise NotImplementedError('tmp_num is zero!!')
            else:
                h = graph.srcdata['feat'][-tmp_num:, :]

                # dst_data = graph.dstdata['feat']
                # src_data = graph.srcdata['feat']
                model = model_list[tmp_model_num]
                h = model(blocks, h, lpa_labels, embedding_flag='feature_embedding')

                embedding_list.append(h)
        target_features = blocks[-1].dstdata['feat']
        model = model_list[len(model_list) - 1]
        target_h = model(blocks, target_features, lpa_labels, embedding_flag='feature_embedding')
        # embedding_list.append(target_h)
        src_h = target_h
        for tmp_model_num in range(len(model_list) - 1):
            src_h = torch.vstack((src_h, embedding_list[-tmp_model_num - 1]))
        src_h_list = [src_h]
        # h_src = feat
        # h_dst = h_src[:graph.number_of_dst_nodes()]
        all_q_embedding_list = []
        all_v_embedding_list = []
        for layer_num in range(len(model_list) - 1):
            q_embedding_list = []
            v_embedding_list = []
            for tmp_model_num in range(len(model_list) - 1 - layer_num):
                # graph = blocks[tmp_model_num]
                # for tmp_model_num in range(len(model_list)-layer_num):
                model = model_list[tmp_model_num]
                # src_0 = embedding_list[tmp_model_num-1]
                # src_1 = embedding_list[tmp_model_num]
                src = embedding_list[tmp_model_num]
                q_src, v_src = model(blocks, src, lpa_labels,
                                     embedding_flag='transformer_neighbor',
                                     layer_flag=layer_num)
                q_embedding_list.append(q_src)
                v_embedding_list.append(v_src)

            model = model_list[len(model_list) - 1]
            target_q_src, target_v_src = model(blocks, src_h[:target_features.shape[0], :], lpa_labels,
                                               embedding_flag='transformer_neighbor',
                                               layer_flag=layer_num)
            q_src_h, v_src_h = target_q_src, target_v_src
            for tmp_model_num in range(len(model_list) - 1 - layer_num):
                q_src_h = torch.vstack((q_src_h, q_embedding_list[-tmp_model_num - 1]))
                v_src_h = torch.vstack((v_src_h, v_embedding_list[-tmp_model_num - 1]))
            all_q_embedding_list.append(q_src_h)
            all_v_embedding_list.append(v_src_h)

            src_h = model(blocks=blocks, features=src_h, labels=lpa_labels, embedding_flag='transformer_target',
                          q_src=q_src_h, v_src=v_src_h,
                          layer_flag=layer_num)
            src_h_list.append(src_h)
            embedding_list = []
            for tmp_model_num in range(layer_num + 1, len(model_list) - 1):
                graph = blocks[tmp_model_num]
                tmp_num = int(graph.srcdata['feat'].shape[0] - graph.dstdata['feat'].shape[0])
                if tmp_num == 0:
                    h = []
                else:
                    h = src_h[-tmp_num:, :]
                    embedding_list.append(h)
        model = model_list[len(model_list) - 1]
        logits = model(blocks, src_h, lpa_labels,
                       embedding_flag='Classifier')
        # logits.requires_grad_(True)
        return logits

    def model_backward(self, train_loss, optimizer_list, lr_scheduler_list):
        for tmp_optimizer_num in optimizer_list:
            tmp_optimizer = optimizer_list[tmp_optimizer_num]
            tmp_optimizer.zero_grad()
        train_loss.backward()
        for tmp_optimizer_num in optimizer_list:
            tmp_optimizer = optimizer_list[tmp_optimizer_num]

            tmp_optimizer.step()
        for tmp_lr_scheduler_num in lr_scheduler_list:
            tmp_lr_scheduler = lr_scheduler_list[tmp_lr_scheduler_num]
            tmp_lr_scheduler.step()

    def build_node_model(self, feat_df, args, device, train_idx, model_num=1000):
        """
        :param feat_df:
        :param args:
        :param device:
        :param train_idx:
        :param model_num:
        :return:
        """
        model_list = {}
        model = DecentralizedGraphAttnModel(in_feats=feat_df.shape[1],
                                            hidden_dim=args['hid_dim'] // 4,
                                            n_classes=2,
                                            heads=[4] * args['n_layers'],  # [4,4,4]
                                            activation=nn.PReLU(),
                                            n_layers=args['n_layers'],
                                            drop=args['dropout'],
                                            device=device,
                                            ref_df=feat_df.iloc[train_idx],
                                            cat_features={}
                                            ).to(device)
        for idx in range(model_num):
            print('idx: {}'.format(idx))
            model_list[idx] = copy.deepcopy(model)

        return model_list


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features={},
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers (the numer of the layered neighbors)
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
        :param device: where to train model
        """

        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(
                nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                              nn.BatchNorm1d(
                                  self.hidden_dim * self.heads[-1]),
                              nn.PReLU(),
                              nn.Dropout(self.drop),
                              nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                                         self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features
        """
        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h

        h = self.layers[1](h)
        h = self.layers[3](h)
        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l + 4](blocks[l], h))
        logits = self.layers[-1](h)

        return logits


class DecentralizedGraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features={},
                 nei_features=None,
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers (the numer of the layered neighbors)
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
        :param device: where to train model
        """

        super(DecentralizedGraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes + 1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

        self.layers.append(DenTargetTransformerConv(in_feats=self.in_feats,
                                                    out_feats=self.hidden_dim,
                                                    num_heads=self.heads[0],
                                                    skip_feat=skip_feat,
                                                    gated=gated,
                                                    layer_norm=layer_norm,
                                                    activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(DenTargetTransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                                        out_feats=self.hidden_dim,
                                                        num_heads=self.heads[l],
                                                        skip_feat=skip_feat,
                                                        gated=gated,
                                                        layer_norm=layer_norm,
                                                        activation=self.activation))

        self.layers.append(DenNeighborTransformerConv(in_feats=self.in_feats,
                                                      out_feats=self.hidden_dim,
                                                      num_heads=self.heads[0],
                                                      ))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(DenNeighborTransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                                          out_feats=self.hidden_dim,
                                                          num_heads=self.heads[l],
                                                          ))
        if post_proc:
            self.layers.append(
                nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                              nn.BatchNorm1d(
                                  self.hidden_dim * self.heads[-1]),
                              nn.PReLU(),
                              nn.Dropout(self.drop),
                              nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                                         self.heads[-1], self.n_classes))

    def feature_embedding(self, features, labels, n2v_feat=None):
        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h

        h = self.layers[1](h)  #
        h = self.layers[3](h)
        return h

    def transformer_target(self, blocks, q_src, v_src, features, layer_num):
        h = features
        # for l in range(self.n_layers):
        l = layer_num
        #     def forward(self, graph, q_src, v_src, feat, get_attention=False):
        h = self.output_drop(self.layers[l + 4](blocks[l], q_src, v_src, h))
        return h

    def transformer_neighbor(self, blocks, features, layer_num):
        h = features
        prefix_layer_num = 4 + self.n_layers
        l = layer_num

        # for l in range(self.n_layers):
        q_src, v_src = self.layers[l + prefix_layer_num](blocks[l], h)
        return q_src, v_src

    def forward(self, blocks, features, labels, n2v_feat=None, embedding_flag='feature_embedding', layer_flag=0,
                q_src=None, v_src=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features
        """
        h = features
        if embedding_flag == 'feature_embedding':
            h = self.feature_embedding(h, labels, n2v_feat)
            return h
        elif embedding_flag == 'transformer_target':
            assert q_src is not None and v_src is not None, print('q_src and v_src should not to be None!')
            h = self.transformer_target(blocks, q_src, v_src, features=h, layer_num=layer_flag)
            return h

        elif embedding_flag == 'transformer_neighbor':
            q_src, v_src = self.transformer_neighbor(blocks, features=h, layer_num=layer_flag)
            return q_src, v_src
        else:
            logits = self.layers[-1](h)
            return logits


if __name__ == '__main__':
    model_build = ModelBuildingDecentralize()
