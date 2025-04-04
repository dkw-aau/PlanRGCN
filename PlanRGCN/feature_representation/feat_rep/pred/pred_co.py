from collections import deque
import json
import os
import pickle
import networkx as nx
from pyvis.network import Network
import itertools
import time
from feat_rep.config import config

class PredicateCommunityCreator:
    def __init__(
        self,
        verbose=True,
        ks=[10, 20, 30, 40, 50, 60, 100, 150, 200],
        save_dir="/PlanRGCN/data/pred/pred_co",
        predicate_requests= '/PlanRGCN/data/pred/batch_response'
    ) -> None:
        self.verbose = verbose
        self.ks = ks
        self.save_dir = save_dir

    def get_predicate_list(self, filepath: str):
        edge_lst = list()
        data = json.load(open(filepath, "r"))
        data = data["results"]["bindings"]
        for bind in data:
            edge_lst.append((bind["p1"]["value"], bind["p2"]["value"]))
        return edge_lst

    def get_predicate_from_dir(self, dir: str):
        files = sorted([x for x in os.listdir(dir) if x.endswith(".json")])
        edge_lst = list()
        nodes = set()
        for f in files:
            temp_edges = self.get_predicate_list(f"{dir}{f}")
            for n1, n2 in temp_edges:
                nodes.add(n1)
                nodes.add(n2)
            edge_lst.extend(temp_edges)
        return edge_lst

    def create_pred_graph(self, edge_lst):
        pred_graph = nx.DiGraph(edge_lst)
        return pred_graph

    # works with directed grpahs
    def get_louvain_communities(
        self,
        dir: str = "/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/",
        save_pred_graph=None,
    ):
        edges = self.get_predicate_from_dir(dir)
        pred_graph = self.create_pred_graph(edges)
        if isinstance(save_pred_graph, str):
            with open(save_pred_graph, "wb") as f_pred:
                pickle.dump(pred_graph, f_pred)
        commun = nx.algorithms.community.louvain_communities(pred_graph)
        commun_2 = []
        for x in commun:
            commun_2.append(list(x))
        path = self.save_dir + "/communities_louvain.pickle"
        with open(path, "wb") as f:
            pickle.dump(commun_2, f)

    def components_preds(self, pred_graph):
        components = nx.strongly_connected_components(pred_graph)
        com_sets = set()
        for i, c in enumerate(components):
            print(f"comp enum: {i:,.0f}")
            com_sets.add(tuple(c))

        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            print(f"{idx:,.0f}")
            for pred in list(com):
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]

        pred2index = pred_2_index
        max_clusters = len(com_sets)
        path = f"{self.save_dir}/components/comps_{str(max_clusters)}.json"
        os.system("mkdir -p " + f"{self.save_dir}/components/")
        with open(path, "w") as f:
            json.dump((pred2index, max_clusters), f)
        return pred2index, max_clusters

    def community_k(self, pred_graph, k=10):
        """_summary_

        Args:
            pred_graph (nx.Graph): Predicate graph
            k (int, optional): the amount of communities to consider. Defaults to 10.
        """
        communities = nx.community.girvan_newman(pred_graph)
        # print(next(communities))
        com_sets = []
        # com_sets = None
        for com in itertools.islice(communities, k):
            extracted_coms = tuple(sorted(c) for c in com)
            for no_com, extr_c in enumerate(extracted_coms):
                if self.verbose:
                    print(f"Working on Commnity {no_com}", end="\r")

                if not extr_c in com_sets:
                    com_sets.append(extr_c)
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Cluster Postprocessing" + "\n")
            print("-" * 40)
        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            for pred in com:
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]

        pred2index = pred_2_index
        max_clusters = len(com_sets)
        return pred2index, max_clusters

    def create_clusters(self, predicate_dir: str, is_component=True):
        start = time.time()
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Predicate Graph Construction" + "\n")
            print("-" * 40)
        # pred_graph_path = f"{self.save_dir}/pred_graph.pickle"
        pred_graph_path = f"{self.save_dir}/pred_graph2.pickle"
        if os.path.exists(pred_graph_path):
            with open(pred_graph_path, "rb") as f:
                pred_graph = pickle.load(f)
        else:
            edge_lst = self.get_predicate_from_dir(predicate_dir)
            pred_graph = self.create_pred_graph(edge_lst)
            with open(pred_graph_path, "wb") as f:
                pred_graph = pickle.dump(pred_graph, f)

        print(f"Predicate Graph constructed after: {time.time()-start:,.2f}")
        if is_component:
            self.components_preds(pred_graph)
            return
        # print(f"Amount of component in input graph: {nx.number_connected_components(pred_graph)}")
        # exit()
        """if save_pred_graph_png != None:
            net = Network()
            net.from_nx(pred_graph)
            net.save_graph(save_pred_graph_png)"""
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Clustering" + "\n")
            print("-" * 40)
        for k in self.ks:
            pred2index, max_clusters = self.community_k(pred_graph, k=k)
            path = f"{self.save_dir}/{str(k)}/community_{str(k)}.json"
            os.system("mkdir -p " + path)
            with open(path, "w") as f:
                json.dump((pred2index, max_clusters), f)


def create_louvain_to_p_index(
    path="/PlanRGCN/data/pred/pred_co/communities_louvain.pickle",
    output_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
):
    if not os.path.exists(path):
        raise Exception("File does not exist " + path)
    cs = pickle.load(open(path, "rb"))
    pred2index = {}
    m = len(cs) + 1
    for i, c in enumerate(cs):
        for p in list(c):
            pred2index[p] = i
    print(pred2index)
    print(m)
    pickle.dump((pred2index, m), open(output_path, "wb"))


def create_k_clique(pred_graph_path=None, pred_graph: nx.DiGraph = None, k=20):
    """crates k clique based pred2com dict. The process gets killed when doing this.

    Args:
        pred_graph_path (str, optional): path to predicate graph on pickle form
        pred_graph (nx.DiGraph, optional): predicate graph
        k (int, optional): the size of the smallest clique. Defaults to 20.

    Raises:
        Exception: if no predicate graph is provided.
    """
    if pred_graph_path == None and pred_graph == None:
        raise Exception(
            "One of the fields 'pred_graph_path' or 'pred_graph' must be specified!"
        )
    if pred_graph is None:
        pred_graph = pickle.load(open(pred_graph_path, "rb"))
    pred_graph = pred_graph.to_undirected()
    c = list(nx.community.k_clique_communities(pred_graph, 20))


def create_kernighan_lin(
    pred_graph_path=None,
    pred_graph: nx.DiGraph = None,
    iterations=10,
    seed=42,
    save_dict_path=None,
):
    """crates kerlinghan lin partition based pred2com dict.

    Args:
        pred_graph_path (str, optional): path to predicate graph on pickle form
        pred_graph (nx.DiGraph, optional): predicate graph
        k (int, optional): the size of the smallest clique. Defaults to 20.

    Raises:
        Exception: if no predicate graph is provided.
    """
    if pred_graph_path == None and pred_graph == None:
        raise Exception(
            "One of the fields 'pred_graph_path' or 'pred_graph' must be specified!"
        )
    if save_dict_path == None:
        raise Exception("'save_dict_path' must be specified!")
    if pred_graph is None:
        pred_graph = pickle.load(open(pred_graph_path, "rb"))
    coms = []

    pred_graph = pred_graph.to_undirected()
    # gaph_stack = deque()
    # graph_stack.append(pred_graph)
    graph_stack = [pred_graph]
    d = 0
    while d != iterations:
        new_stack = list()
        for p in graph_stack:
            partitions = nx.community.kernighan_lin.kernighan_lin_bisection(
                p, seed=seed
            )
            partitions = [list(x) for x in partitions]
            left = nx.Graph.subgraph(p, partitions[0])
            right = nx.Graph.subgraph(p, partitions[1])
            if len(left.nodes()) > 1:
                new_stack.append(left)
            if len(right.nodes()) > 1:
                new_stack.append(right)
            coms.extend(partitions)
        graph_stack = new_stack
        d += 1

    pred2idx = {}
    for n in pred_graph.nodes:
        pred2idx[n] = list()
    for idx, c in enumerate(coms):
        for c_pred in c:
            pred2idx[c_pred].append(idx)
    m = len(coms) + 1
    with open(save_dict_path, "wb") as f_path:
        pickle.dump((pred2idx, m), f_path)

def louvain_feat_const(config):
    """Using the previously extracted predicate pair ({p1,p2 | <s,p1,s2>, <s2, p2, s3>}}), we first construct a predicate graph 
    and then proceed to construct predicate to integer mapping by decomposing the graph into 
    """
    f = open(config['time_log'], 'a')
    d = PredicateCommunityCreator(save_dir=config['save_dir'])
    t = time.time()
    d.get_louvain_communities(
        dir=config['pred_co_pair_dir']
    )
    dur = time.time() - t
    f.write(f"Louvain Communication, {dur}\n")
    f.flush()
    t = time.time()
    create_louvain_to_p_index(
        path=config["comm"],
        output_path=config["p2i"],
    )
    dur = time.time()-t
    f.write(f"pred2index, {dur}\n")
    f.flush()
    f.close()

if __name__ == "__main__":
    
    #config = {
    #    'save_dir' : '/PlanRGCN/data/wikidata/predicate/pred_co',
    #    'pred_co_pair_dir' : '/PlanRGCN/data/wikidata/predicate/batch_response/',
    #}
    #config["comm"] = f"{config['save_dir']}/communities_louvain.pickle"
    #config["p2i"] = f"{config['save_dir']}/pred2index_louvain.pickle"
    #config["time_log"] = f"{config['save_dir']}/time_pred_co_louvain.txt"
    
    louvain_feat_const(config)
    
    
    
    exit()
    d = PredicateCommunityCreator(save_dir="/PlanRGCN/data/pred/pred_co")
    t = time.time()
    d.get_louvain_communities(
        dir="/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/"
    )
    dur = time.time() - t
    t = time.time()
    create_louvain_to_p_index(
        path="/PlanRGCN/data/pred/pred_co/communities_louvain.pickle",
        output_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    )
    dur = time.time()-t
    
    exit()
    """Girvan-newman and other clustering approach, but these are too time consuming.
    """
    d.create_clusters(
        "/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/"
    )
    exit()
