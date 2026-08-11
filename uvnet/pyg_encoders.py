"""PyTorch Geometric implementation of the UV-Net encoders."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_max_pool


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError("Number of layers should be positive")
        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
            return

        self.linear_or_not = False
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
        )

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        hidden = x
        for index in range(self.num_layers - 1):
            hidden = F.relu(self.batch_norms[index](self.linears[index](hidden)))
        return self.linears[-1](hidden)


class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} curve channels, found {x.size(1)}")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x).flatten(1)
        return self.fc(x)


class UVNetSurfaceEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dims=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x):
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} surface channels, found {x.size(1)}")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x).flatten(1)
        return self.fc(x)


class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super().__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = nn.Parameter(torch.tensor([0.0]))

    def forward(self, edge_index, node_features, edge_features):
        source, destination = edge_index
        endpoint_features = self.proj(node_features[source]) + self.proj(
            node_features[destination]
        )
        hidden = self.mlp((1 + self.eps) * edge_features + endpoint_features)
        return F.leaky_relu(self.batchnorm(hidden))


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        super().__init__()
        edge_network = nn.Linear(edge_feats, node_feats * out_feats)
        self.gconv = NNConv(
            in_channels=node_feats,
            out_channels=out_feats,
            nn=edge_network,
            aggr="add",
            root_weight=False,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = nn.Parameter(torch.tensor([0.0]))

    def forward(self, edge_index, node_features, edge_features):
        hidden = (1 + self.eps) * node_features
        hidden = self.gconv(hidden, edge_index, edge_features)
        hidden = self.mlp(hidden)
        return F.leaky_relu(self.batchnorm(hidden))


class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.node_conv_layers = nn.ModuleList()
        self.edge_conv_layers = nn.ModuleList()

        for layer in range(num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats,
                    hidden_dim,
                    edge_feats,
                    num_mlp_layers,
                    hidden_dim,
                )
            )
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats,
                    hidden_dim,
                    node_feats,
                    num_mlp_layers,
                    hidden_dim,
                )
            )

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            layer_dim = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(layer_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)

    def forward(self, edge_index, batch, node_features, edge_features):
        hidden_representations = [node_features]
        hidden_nodes = node_features
        hidden_edges = edge_features

        for layer in range(self.num_layers - 1):
            hidden_nodes = self.node_conv_layers[layer](
                edge_index, hidden_nodes, hidden_edges
            )
            hidden_edges = self.edge_conv_layers[layer](
                edge_index, hidden_nodes, hidden_edges
            )
            hidden_representations.append(hidden_nodes)

        node_output = self.drop1(hidden_representations[-1])
        graph_output = 0
        for layer, hidden in enumerate(hidden_representations):
            pooled = global_max_pool(hidden, batch)
            graph_output = graph_output + self.drop(
                self.linears_prediction[layer](pooled)
            )
        return node_output, graph_output
