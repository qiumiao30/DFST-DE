import torch
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, in_feat, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_feat  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.time_step = in_feat
        self.in_feat = in_feat
        # self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(self.in_feat, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化
        self.weight_key = nn.Parameter(torch.zeros(size=(self.in_feat, 1)))  # nn.Parameter:参数初始化
        self.weight_query = nn.Parameter(torch.zeros(size=(self.in_feat, 1)))

        embed_dim = 128
        self.embedding = nn.Embedding(in_features, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # self.GRU = nn.GRU(self.time_step, self.in_features, batch_first=True)  ##GRU input: (time_step, hidden(features))

    # def self_graph_attention(self, input):
    #     input = input.contiguous()
    #     bat, N, fea = input.size()
    #     key = torch.matmul(input, self.weight_key)
    #     query = torch.matmul(input, self.weight_query)
    #     data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1) # repeat()重复
    #     data = data.squeeze(2) # 减少第二个维度，如果第二个维度为1的话
    #     data = data.view(bat, N, -1)
    #     data = self.leakyrelu(data)
    #     attention = F.softmax(data, dim=2)
    #     attention = self.dropout(attention)
    #     return attention

    def forward(self, x):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        # input, _ = self.GRU(x.permute(2, 0, 1).contiguous())  ##contiguous(),改变数据存储地址
        # input = input.permute(1, 0, 2).contiguous()  ##permute()不会改变存储位置
        # attention = self.self_graph_attention(input)
        # attention = torch.mean(attention, dim=0)
        # # degree = torch.sum(attention, dim=1)
        # # laplacian is sym or not
        # adj = 0.5 * (attention + attention.T)

        all_embeddings = self.embedding(torch.arange(38))
        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(38, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        adj = cos_ji_mat / normed_mat

        top_k = 10
        filter_value = float(1)
        indices_to_remove = adj > torch.topk(adj, top_k)[0][..., -2, None]
        # print(indices_to_remove)
        adj[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        x = x.permute(0, 2, 1)
        h = torch.matmul(x, self.W)  # [N, out_features]
        N = h.size()[1]  # N 图的节点数

        left = h.repeat_interleave(N, dim=1)
        right = h.repeat(1, N, 1)
        a_input = torch.cat((left, right), dim=2).reshape(h.shape[0], N, N, -1)

        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj == 1, e, zero_vec)  # [N, N]
        print(attention)
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        # attention = F.softmax(e, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        print("iiiii", attention)
        attention = F.dropout(attention, 0.2, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime).permute(0,2,1)
        else:
            return h_prime.permute(0,2,1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, n_features, in_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_features, in_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_features, n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        # ConsatntPad1d(padding,value)-->padding可以为int或tuple。
        # input:(N,C,Win),output:(N,C,Wout),Wout=Win+padding_left+padding_right
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0) # 左右两边添加padding个常数
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # input:(N, Cin, L)-->(batch,channel,seq_len), x:(batch,seq_len,channel)
        x = self.padding(x)
        x = self.relu(self.conv(x)) # output:(N, Cout, Lout)-->(batch, channel, seq_len)
        return x.permute(0, 2, 1)  # Permute back

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     图注意力层
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features  # 节点表示向量的输入特征维度
#         self.out_features = out_features  # 节点表示向量的输出特征维度
#         self.time_step = in_features
#         # self.dropout = dropout  # dropout参数
#         self.alpha = alpha  # leakyrelu激活的参数
#         self.concat = concat  # 如果为true, 再进行elu激活
#
#         # 定义可训练参数，即论文中的W和a
#         self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化
#         self.weight_key = nn.Parameter(torch.zeros(size=(self.in_features, 1)))  # nn.Parameter:参数初始化
#         self.weight_query = nn.Parameter(torch.zeros(size=(self.in_features, 1)))
#
#         self.dropout = nn.Dropout(dropout)
#
#         # 定义leakyrelu激活函数
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#         self.GRU = nn.GRU(self.time_step, self.in_features, batch_first=True)  ##GRU input: (time_step, hidden(features))
#
#     def self_graph_attention(self, input):
#         input = input.contiguous()
#         bat, N, fea = input.size()
#         key = torch.matmul(input, self.weight_key)
#         query = torch.matmul(input, self.weight_query)
#         data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1) # repeat()重复
#         data = data.squeeze(2) # 减少第二个维度，如果第二个维度为1的话
#         data = data.view(bat, N, -1)
#         data = self.leakyrelu(data)
#         attention = F.softmax(data, dim=2)
#         attention = self.dropout(attention)
#         return attention
#
#     def forward(self, x):
#         """
#         inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
#         adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
#         """
#         input, _ = self.GRU(x.permute(2, 0, 1).contiguous())  ##contiguous(),改变数据存储地址
#         input = input.permute(1, 0, 2).contiguous()  ##permute()不会改变存储位置
#         attention = self.self_graph_attention(input)
#         attention = torch.mean(attention, dim=0)
#         # degree = torch.sum(attention, dim=1)
#         # laplacian is sym or not
#         adj = 0.5 * (attention + attention.T)
#
#         topk_num = 10
#         topk_indices_ji = torch.topk(adj, topk_num, dim=-1)[1]
#         self.learned_graph = topk_indices_ji
#
#         gated_i = torch.arange(0, x.shape[2]).T.unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0)
#         gated_j = topk_indices_ji.flatten().unsqueeze(0)
#         gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
#
#         x = x.permute(0, 2, 1)
#         h = torch.matmul(x, self.W)  # [N, out_features]
#         N = h.size()[1]  # N 图的节点数
#
#         left = h.repeat_interleave(N, dim=1)
#         right = h.repeat(1, N, 1)
#         a_input = torch.cat((left, right), dim=2).reshape(h.shape[0], N, N, -1)
#
#         # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         # [N, N, 2*out_features]
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
#         # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
#
#         zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
#         attention = torch.where(adj > e.mean(), e, zero_vec)  # [N, N]
#         # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
#         # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
#         attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
#         attention = F.dropout(attention, 0.2, training=self.training)  # dropout，防止过拟合
#         h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
#         # 得到由周围节点通过注意力权重进行更新的表示
#         if self.concat:
#             return F.elu(h_prime).permute(0,2,1)
#         else:
#             return h_prime.permute(0,2,1)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
# class GAT(nn.Module):
#     def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
#         """Dense version of GAT
#         n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
#         从不同的子空间进行抽取特征。
#         """
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         # 定义multi-head的图注意力层
#         self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
#         # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
#         self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x):
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = torch.cat([att(x) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = F.elu(self.out_att(x))  # 输出并激活
#         return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定

Tensor = torch.Tensor
def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, embed_dim, alpha, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.zeros((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        embed_dim = self.embed_dim
        self.embedding = nn.Embedding(n_features, embed_dim)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        #==================================================

        all_embeddings = self.embedding(torch.arange(self.n_features))

        weights_arr = all_embeddings.detach().clone()
        weights = weights_arr.view(self.n_features, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        learned_graph = cos_ji_mat / normed_mat

        norm = torch.norm(all_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.transpose(0, 1))

        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.

        learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)
        adj = gumbel_softmax(learned_graph, tau=1, hard=True)

        adj = adj[:, :, 0].clone().reshape(self.n_features, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.n_features, self.n_features).bool()
        adj = adj.masked_fill_(mask, 0)
        #==================================================

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

            #############################################
            zero_vec = -1e12 * torch.ones_like(e)
            e = torch.where(adj == 1, e, zero_vec)  # [N, N]
            #############################################

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        # out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

import math
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionBlock(nn.Module):
  """An attention mechanism similar to Vaswani et al (2017)
  The input of the AttentionBlock is `BxTxD` where `B` is the input
  minibatch size, `T` is the length of the sequence `D` is the dimensions of
  each feature.
  The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
  attention values.
  Arguments:
      dims (int): the number of dimensions (or channels) of each element in
          the input sequence
      k_size (int): the size of the attention keys
      v_size (int): the size of the attention values
      seq_len (int): the length of the input and output sequences
  """
  def __init__(self, dims, k_size, v_size, seq_len=None):
    super(AttentionBlock, self).__init__()
    self.key_layer = nn.Linear(dims, k_size)
    self.query_layer = nn.Linear(dims, k_size)
    self.value_layer = nn.Linear(dims, v_size)
    self.sqrt_k = math.sqrt(k_size)

  def forward(self, minibatch):
    keys = self.key_layer(minibatch)
    queries = self.query_layer(minibatch)
    values = self.value_layer(minibatch)
    logits = torch.bmm(queries, keys.transpose(2,1))
    # Use numpy triu because you can't do 3D triu with PyTorch
    # TODO: using float32 here might break for non FloatTensor inputs.
    # Should update this later to use numpy/PyTorch types of the input.
    mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
    mask = torch.from_numpy(mask).cuda()
    # do masked_fill_ on data rather than Variable because PyTorch doesn't
    # support masked_fill_ w/-inf directly on Variables for some reason.
    logits.data.masked_fill_(mask, float('-inf'))
    probs = F.softmax(logits, dim=1) / self.sqrt_k
    read = torch.bmm(probs, values)
    return minibatch + read

class TCN(nn.Module):
    def __init__(self, input_size, dim_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], dim_size)
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1.permute(0, 2, 1))
        return o
