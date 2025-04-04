import dgl
import dgl.nn.pytorch as dglnn
from graph_construction.query_graph import QueryPlan, QueryPlanCommonBi
import torch.nn as nn
import torch.nn.functional as F
import torch


class Classifier2RGCN(nn.Module):
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
        
class Classifier2SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, p, n_classes, aggregate='pool'):
        """

        Args:
            in_dim (int): _description_
            hidden_dim1 (int): _description_
            hidden_dim2 (int): _description_
            p (float): _description_
            n_classes (int): _description_
        """
        super(Classifier2SAGE, self).__init__()
        self.conv1 = dglnn.SAGEConv(
            in_dim, hidden_dim1,aggregate , feat_drop=p
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.SAGEConv(
            hidden_dim1, hidden_dim2, aggregate, feat_drop=p
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
        h = self.conv1(g, h)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return F.softmax(self.classify(hg), dim=1)
        
class ClassifierRGCNHidden(nn.Module):
    def __init__(self, config={'in_dim':0, 'rel_layers':[100,200,100], 'hidden':[100,200,100], 'n_classes':3}, p=0.5):
        """Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations

        Args:
            config:
                in_dim: input dimention
                rel_layers : list of relation layers.
                hidden : hidden layers.
                n_classes : number of time intervals to predict between.
            in_dim (int): _description_
            hidden_dim1 (int): _description_
            hidden_dim2 (int): _description_
            p (float): _description_
        """
        super(ClassifierRGCNHidden, self).__init__()
        prev = config['in_dim']
        self.rel_convs = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()
        for i in config['rel_layers']:
            self.rel_convs.append(dglnn.RelGraphConv(
                prev, i, QueryPlan.max_relations, dropout=p)
            )
            prev = i
            
        for i in config['hidden']:
            self.hidden_layers.append(nn.Linear(
                prev, i)
            )
            prev = i
        
        self.classify = nn.Linear(prev, config['n_classes'])
        self.in_dim = config['in_dim']
        self.config = config
        self.n_classes = config['n_classes']

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        for l in self.rel_convs:
            h = l(g,h,rel_types)
            h = F.relu(h)
        with g.local_scope():
            g.ndata["node_features"] = h
            hg = dgl.mean_nodes(g, "node_features")
            for l in self.hidden_layers:
                hg = F.relu(l(hg),)
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
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        for l in self.rel_convs:
            h = l(g,h,rel_types)
            h = F.relu(h)
        with g.local_scope():
            g.ndata["node_features"] = h
            hg = dgl.mean_nodes(g, "node_features")
            for l in self.hidden_layers:
                hg = F.relu(l(hg),)
            return hg

    def get_dims(self):
        return self.config


class Classifier2RGCNAuto(nn.Module):
    def __init__(
        self, in_dim, auto_layer1, auto_layer2, hidden_dim1, hidden_dim2, p, n_classes
    ):
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
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            return dgl.mean_nodes(g, "node_features")

    def get_dims(self):
        return self.in_dim, self.hidden_dim1, self.hidden_dim2, self.n_classes


class Classifier(nn.Module):
    """_summary_
    Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

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


class ClassifierGridSearch(nn.Module):
    """_summary_
    Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_dim, layers, n_classes, auto_neurons=None, add_dropout=False):
        """_summary_

        Args:
            in_dim (int): _description_
            layers (list[tuple]): _description_
            n_classes (int): _description_
        """
        super().__init__()
        self.is_auto = True if auto_neurons != None else False
        self.rgcn_neurons = {}
        self.fc_neurons = {}
        self.layers = nn.ModuleList()
        self.auto_neurons = auto_neurons
        prev = in_dim

        self.add_dropout = add_dropout
        self.rgcn_last_idx = 0
        self.contains_fc = False

        if self.is_auto:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, auto_neurons, dtype=torch.float32), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(auto_neurons, in_dim, dtype=torch.float32)
            )
            prev = auto_neurons
        for layer_no, (t, neurons) in enumerate(layers):
            if t == 1:
                if self.add_dropout:
                    self.layers.append(
                        dglnn.RelGraphConv(
                            prev, neurons, QueryPlan.max_relations, dropout=0.5
                        )
                    )
                else:
                    self.layers.append(
                        dglnn.RelGraphConv(prev, neurons, QueryPlan.max_relations)
                    )
                self.rgcn_neurons[f"RGCN {layer_no}"] = neurons
                # self.layers.append(nn.Dropout(p=0.5))
                self.rgcn_last_idx = layer_no
            elif t == 2:
                self.contains_fc = True
                # fully connected layers (optional)
                self.layers.append(nn.Linear(prev, neurons))
                self.fc_neurons[f"fc {layer_no}"] = neurons
            prev = neurons

        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(prev, n_classes)
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.model_str = self.make_model_str()

    def forward(self, g, h, rel_types):
        if h.dtype != torch.float32:
            h = h.type(torch.float32)

        decoded = None
        if self.is_auto:
            h = self.encoder(h)
            decoded = self.decoder(h)
        for layer_no, l in enumerate(self.layers):
            if layer_no <= self.rgcn_last_idx:
                h = l(g, h, rel_types)
                h = F.relu(h)
            elif layer_no == (self.rgcn_last_idx + 1) and self.contains_fc:
                with g.local_scope():
                    g.ndata["node_features"] = h
                    # Calculate graph representation by average readout.
                    h = dgl.mean_nodes(g, "node_features")
                    h = F.relu(l(h))
            elif layer_no > (self.rgcn_last_idx + 1) and self.contains_fc:
                h = F.relu(l(h))
        if not self.contains_fc:
            with g.local_scope():
                g.ndata["node_features"] = h
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, "node_features")
                if self.is_auto:
                    return decoded, F.softmax(self.classify(hg), dim=1)
                return F.softmax(self.classify(hg), dim=1)
        else:
            if self.is_auto:
                return decoded, F.softmax(self.classify(h), dim=1)
            return F.softmax(self.classify(h), dim=1)

    def get_dims(self):
        return self.in_dim, self.rgcn_neurons, self.fc_neurons, self.n_classes

    def make_model_str(self):
        if self.is_auto:
            t = f"auto_{self.auto_neurons}_"
        else:
            t = ""
        for idx, rgcn_label in enumerate(self.rgcn_neurons.keys()):
            t += f"RGCN_{idx}_{self.rgcn_neurons[rgcn_label]}_"
        for fc_label in self.fc_neurons.keys():
            t += f"fc{idx}_{self.fc_neurons[fc_label]}_"
        return t


class ClassifierWAuto(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierWAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=torch.float32), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, in_dim, dtype=torch.float32))
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.dropout2 = nn.Dropout(0.2)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.encoder(h)
        decoded = self.decoder(h)
        h = self.dropout1(h)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        h = self.dropout2(h)
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return decoded, F.softmax(self.classify(hg), dim=1)


class ClassifierWSelfTriple(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierWSelfTriple, self).__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

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


class RegressorWSelfTriple(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(RegressorWSelfTriple, self).__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 1)

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
            return F.relu(self.classify(hg))

if __name__ == "__main__":
    import networkx as nx
    import numpy as np
    
    edge_list = [(0,1),(0,2),(0,3),(0,4),(0,5) ]
    edge_type = np.array([0,2,0,0,4], dtype=np.float32)
    edge_type = torch.tensor(edge_type)
    nodes = [x for x in range(10)]
    node_feats = torch.tensor(np.random.sample([len(nodes),10]))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    G = dgl.from_networkx(G)
    #test code
    config={'in_dim':10, 'rel_layers':[100,200,100], 'hidden':[], 'n_classes':3}
    p=0.5
    cls = ClassifierRGCNHidden(config=config, p=p)
    print(cls(G,node_feats, edge_type))