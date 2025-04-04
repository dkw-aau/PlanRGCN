import dgl
import dgl.nn.pytorch as dglnn
from graph_construction.query_graph import QueryPlan, QueryPlanCommonBi
import torch.nn as nn
import torch.nn.functional as F
import torch


class Regression(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, p, n_classes):
        """Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations

        Args:
            in_dim (int): _description_
            hidden_dim1 (int): _description_
            hidden_dim2 (int): _description_
            p (float): _description_
            n_classes (int): _description_
        """
        super(Classifier2RGCN, self).__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim1, QueryPlan.max_relations, dropout=p
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim1, hidden_dim2, QueryPlan.max_relations, dropout=p
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim2, n_classes)
        self.in_dim = in_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_classes = n_classes

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return F.softmax(self.classify(hg), dim=1)

    def get_last_hidden_layer(self, g, h, rel_types):
        """This method will return the embedding of the last hidden layer from the model which can be used as an representation of the query plan in a downstream task

        Args:
            g (dgl.DGLGraph): query plan graph
            h (torch.Tensor): the node features
            rel_types (torch.Tensor): Tensor of the relation type for each edge in the graph

        Returns:
            torch.Tensor: Latent embedding of the query plan.
        """
def load_state_dict_helper(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

