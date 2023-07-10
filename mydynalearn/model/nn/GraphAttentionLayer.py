import torch
import torch.nn as nn
from torch import Tensor
from mydynalearn.model.util.MultiHeadLinear import MultiHeadLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class GraphAttentionLayer(MessagePassing):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=False,
        bias=True,
        edge_in_channels=0,
        edge_out_channels=0,
        self_attention=True,
        normalize=True,
        **kwargs,
    ):
        super(GraphAttentionLayer, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.bias = bias
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.self_attention = self_attention
        self.normalize = normalize
        self._alpha = None
        self.activation = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.2)

        if isinstance(in_channels, int):
            self.linear_source = nn.Linear(in_channels, heads * out_channels, bias=bias)
            self.linear_target = nn.Linear(in_channels, heads * out_channels, bias=bias)
        else:
            self.linear_source = nn.Linear(
                in_channels[0], heads * out_channels, bias=bias
            )
            self.linear_target = nn.Linear(
                in_channels[1], heads * out_channels, bias=bias
            )

        if edge_in_channels > 0 and edge_out_channels > 0:
            self.linear_edge = nn.Linear(
                edge_in_channels, heads * edge_out_channels, bias=bias
            )
            self.edge_combine = nn.Linear(
                (edge_out_channels + 1) * heads,
                edge_out_channels * heads,
                bias=bias,
            )
            self.attn_edge = MultiHeadLinear(edge_out_channels, heads=heads, bias=bias)

        else:
            self.register_parameter("linear_edge", None)
            self.register_parameter("edge_combine", None)
            self.register_parameter("attn_edge", None)

        self.attn_source = MultiHeadLinear(out_channels, heads=heads, bias=bias)
        self.attn_target = MultiHeadLinear(out_channels, heads=heads, bias=bias)

        if self_attention:
            self.self_attn_source = MultiHeadLinear(
                out_channels, heads=heads, bias=bias
            )
            self.self_attn_target = MultiHeadLinear(
                out_channels, heads=heads, bias=bias
            )
        else:
            self.register_parameter("self_attn_source", None)
            self.register_parameter("self_attn_target", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear_source.weight)
        zeros(self.linear_source.bias)
        glorot(self.linear_target.weight)
        zeros(self.linear_target.bias)
        if self.linear_edge is not None:
            glorot(self.linear_edge.weight)
            zeros(self.linear_edge.bias)
            glorot(self.edge_combine.weight)
            zeros(self.edge_combine.bias)
            self.attn_edge.reset_parameters()

        self.attn_source.reset_parameters()
        self.attn_target.reset_parameters()

        if self.self_attn_source is not None:
            self.self_attn_source.reset_parameters()
        if self.self_attn_target is not None:
            self.self_attn_target.reset_parameters()


    def message(
            self,
            x_target_j,
            alpha_source_i,
            alpha_target_j,
            edge_attn,):
        alpha = alpha_source_i + alpha_target_j  # GAT原文公式(1) e_ij
        # alpha = alpha if edge_attn is None else alpha + edge_attn  # 考虑边，但是这里和deeplearning原文公式(11)不太一样
        # alpha = torch.sigmoid(alpha) # GAT中公式(2) softmax，这里是sigmoid。 在原文中是公式(11)
        alpha = self.leakyrelu(alpha) # GAT中公式(2) softmax，这里是sigmoid。 在原文中是公式(11)
        self._alpha = alpha
        x_target_j = x_target_j.view(-1, self.heads, self.out_channels)
        a_ij = (x_target_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels) # 公式(11) 中括号内的元素。还未激活
        return a_ij

    def forward(self,
                x,
                edge_index,
                edge_attr=None,
                return_attention_weights=False,):
        H, C = self.heads, self.out_channels
        x_s: OptTensor = None
        x_t: OptTensor = None
        alpha_s: OptTensor = None
        alpha_t: OptTensor = None
        alpha_e: OptTensor = None
        self_alpha_s: OptTensor = None
        self_alpha_t: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `DynamicsGATConv`."
            x_s = x * 1
            x_t = x * 1
        else:
            x_s, x_t = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `DynamicsGATConv`."
            if x_t is None:
                x_t = x_s * 1
        # x: 节点状态 (1000,32)
        # x_s == x_t == x   对应GAT原文中的节点特征h

        x_s = self.linear_source(x_s).view(-1, H, C)  # 考虑了att头并进行线性变化 (1000,2,32)。 对应GAT原文公式(1)whi
        x_t = self.linear_target(x_t).view(-1, H, C)  # 考虑了att头并进行线性变化 (1000,2,32)。 对应GAT原文公式(1)whj
        x_s, x_t = self.activation(x_s), self.activation(x_t)
        alpha_s = self.attn_source(x_s)  # ei (1000,2)。 对应GAT原文公式(1)ei  (注意：这个(1000,2) 2是head)
        alpha_t = self.attn_target(x_t)  # ej (1000,2)。 对应GAT原文公式(1)ej
        if self.self_attention:
            self_alpha_s = self.self_attn_source(x_s)
            self_alpha_t = self.self_attn_target(x_t)
        x_s = x_s.view(-1, H * C)  # (1000,64)
        x_t = x_t.view(-1, H * C)  # (1000,64)

        assert x_s is not None
        assert alpha_s is not None

        if self.linear_edge is not None:
            assert edge_attr is not None
            edge_attr = self.linear_edge(edge_attr).view(-1, H, self.edge_out_channels)
            alpha_e = self.attn_edge(edge_attr)

        # propagation
        out = self.propagate(
            edge_index,
            x_source=x_s,
            x_target=x_t,
            alpha_source=alpha_s,
            alpha_target=alpha_t,
            edge_attn=alpha_e,
        ).view(-1, H, C)


        if self.self_attention:
            out += x_s.view(-1, H, C)   # 传播完了加上自身
            # out += torch.sigmoid(self_alpha_t + self_alpha_s).unsqueeze(-1) * x_t.view(
            #     -1, H, C
            # )

        alpha = self._alpha
        self._alpha = None

        # combining edge attributes with attention coefficients
        if self.linear_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.cat([edge_attr, alpha.unsqueeze(-1)], axis=-1)
            edge_attr = edge_attr.view(-1, H * (self.edge_out_channels + 1))
            out_edge = self.edge_combine(edge_attr).view(-1, H, self.edge_out_channels)
        else:
            out_edge = None

        # combining attention heads
        if self.concat:
            out = out.view(-1, H * C)
            if out_edge is not None:
                out_edge = out_edge.view(-1, H * self.edge_out_channels)
        else:
            out = out.mean(dim=1)
            if out_edge is not None:
                out_edge = out_edge.mean(dim=1)
        # return
        if return_attention_weights:
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            if out_edge is not None:
                return out, out_edge
            else:
                return out
        return out
    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
