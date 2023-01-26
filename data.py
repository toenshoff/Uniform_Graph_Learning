import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def star_edges(k):
    src = torch.arange(1, k + 1)
    tgt = torch.zeros((k,), dtype=torch.long)
    edge_index = torch.stack([src, tgt], dim=0)
    edge_index = to_undirected(edge_index)
    return edge_index


def unlabeled_fixed_deg_edges(k, c):
    # get first edge layer
    high_edge_idx = star_edges(k)

    # construct second edge layer
    src = torch.arange(k + 1, k + c + 1).repeat_interleave(k)
    tgt = torch.arange(1, k + 1).repeat(c)
    low_edge_index = torch.stack([src, tgt], dim=0)
    low_edge_index = to_undirected(low_edge_index)

    edge_index = torch.cat([high_edge_idx, low_edge_index], dim=1)
    return edge_index


def unlabeled_random_deg_edges(k, c):
    # get first edge layer
    high_edge_idx = star_edges(k)

    # construct second edge layer
    degree = torch.randint(0, c+1, (k,))
    num_leaves = degree.sum()
    src = torch.arange(k + 1, k + num_leaves + 1)
    tgt = torch.arange(1, k + 1).repeat_interleave(degree)
    low_edge_index = torch.stack([src, tgt], dim=0)
    low_edge_index = to_undirected(low_edge_index)

    edge_index = torch.cat([high_edge_idx, low_edge_index], dim=1)
    return edge_index, degree



def root_label_mask(num_nodes):
    mask = torch.zeros((num_nodes,), dtype=torch.bool)
    mask[0] = True
    return mask


def ucf_data(k, c):
    edge_index = star_edges(k)
    x = torch.tensor([0] + [c] * k, dtype=torch.float32)
    y = torch.zeros((k+1,), dtype=torch.float32)
    y[0] = c
    mask = root_label_mask(k + 1)
    data = Data(x=x, y=y, edge_index=edge_index, mask=mask)
    return data


def svf_data(k, c):
    edge_index = unlabeled_fixed_deg_edges(k, c)
    x = torch.ones((k + c + 1,), dtype=torch.float32)
    y = torch.zeros((k + c + 1,), dtype=torch.float32)
    y[0] = c
    mask = root_label_mask(k + c + 1)
    data = Data(x=x, y=y, edge_index=edge_index, mask=mask)
    return data
