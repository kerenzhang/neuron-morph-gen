from typing import Any, List, Tuple
from ordered_set import OrderedSet
import os
import networkx as nx
import numpy as np


class Graph:

    def __init__(self, bifs: List=None, scheme: str='c') -> None:
        """
        Default in centrifugal branch order (c), can specify in shaft (s)
        :param bifs: list of bifs that contain context info
        """
        self.bifs = bifs
        self.scheme = scheme
        self.conn_pt = {}
        order_list = []
        segment_list = []
        for b in self.bifs:
            order_list.append(b['order'])
            segment_list.extend(b['context']['c'])
        if len(bifs) > 0:
            self.num_segments = np.max(segment_list) + 1
        else:
            self.num_segments = 1
        if order_list:
            self.max_order = np.max(order_list) + 1
        else:
            self.max_order = 1
        self.order_array = []
        self.shaft_order_array = []
        self.reverse_order_array = []
        self.tree_graph = self.get_tree_graph()
        self.order_graph = self.get_order_graph()
        self.term_branches = np.squeeze(np.argwhere(np.sum(self.tree_graph, axis=1) == 0))
        self.G = nx.from_numpy_array(self.tree_graph)
        if self.num_segments > 1:
            self.all_rooted_paths = [np.squeeze(list(nx.all_simple_paths(self.G, source=0, target=t))) for t in self.term_branches]
        else:
            self.all_rooted_paths = []
        if self.scheme == 's':
            self.get_order_graph('s')
        self.get_reverse_order()

    def get_tree_graph(self) -> np.ndarray:
        """
        use adjacency matrix to store the graph (in centrifugal branch order scheme) A[i][j] is the parent i -> child j
        :return:
        """
        am = np.zeros((self.num_segments, self.num_segments), dtype=int)
        for bif in self.bifs:
            p, c = (bif['context']['p'], bif['context']['c'])
            if len(c) > 0:
                for c_i in c:
                    am[p, c_i] = 1
        return am

    def get_order_graph(self, scheme: str='c') -> dict:
        """
        Get the ordered graph based on scheme. Currently two schemes are defined: centrifugal (c) and shaft (s).
        """
        if scheme == 'c':
            order_graph = self.trs_steps()
        elif scheme == 's':
            order_graph = dict()
            for i in range(1, self.max_order + 1):
                order_graph.update({i: set()})
            main_paths = self.all_rooted_paths[np.argmax([len(a) for a in self.all_rooted_paths])]
            order_array = np.zeros(self.num_segments, dtype=int)
            for order, branch in self.trs_steps().items():
                for br in branch:
                    if br in main_paths:
                        order_array[br] = 1
                        order_graph[1].add(br)
                    else:
                        br_p = self.get_parent(br)
                        order_array[br] = order_array[br_p] + 1
                        order_graph[order_array[br_p] + 1].add(br)
            self.shaft_order_array = order_array
            return order_graph
        else:
            order_graph = {}
            print('Scheme {} not pre-defined'.format(scheme))

        return order_graph

    def get_reverse_order(self) -> None:
        self.reverse_order_array = np.ones(self.num_segments, dtype=int)
        for path in self.all_rooted_paths:
            for node in path[:-1]:
                path_to_term = np.squeeze(list(nx.all_simple_paths(self.G, source=node, target=path[-1])))
                dist_2_term = len(path_to_term)
                if dist_2_term > self.reverse_order_array[node]:
                    self.reverse_order_array[node] = dist_2_term

    def get_children(self, idx: int) -> np.ndarray:
        """Return the children."""
        children = np.squeeze(np.argwhere(self.tree_graph[idx, :] == 1))
        if len(children) > 0 and self.get_order(children[1], 'r') > self.get_order(children[0], 'r'):
            children = children[::-1]
        return children

    def get_parent(self, idx: int) -> int:
        """Return the parent."""
        parent_idx = np.argwhere(self.tree_graph[:, idx] == 1)
        if len(parent_idx) > 0:
            parent = parent_idx[0][0]
        else:
            parent = -1
        return parent

    def get_order(self, idx: int, scheme: str='c') -> Any:
        if scheme == 'c':
            return self.order_array[idx]
        elif scheme == 's':
            if self.shaft_order_array is not None:
                self.get_order_graph(scheme='s')
            return self.shaft_order_array[idx]
        elif scheme == 'r':
            return self.reverse_order_array[idx]

    def update_conn(self, children: List, conn: int) -> None:
        for c in children:
            self.conn_pt.update({c: conn})

    def trs_steps(self) -> dict:
        self.order_array = np.zeros(self.num_segments, dtype=int)
        steps = dict()
        for i in range(1, self.max_order + 1):
            steps.update({i: set()})
        if not self.bifs:
            steps[1].add(0)
            self.order_array[0] = 1
        for b in self.bifs:
            p, c = (b['context']['p'], b['context']['c'])
            d = b['order']
            steps[d].add(p)
            self.order_array[p] = d
            for cc in c:
                if cc not in steps[d]:
                    steps[d + 1].add(cc)
                    self.order_array[cc] = d + 1
        return steps

    @property
    def diameter(self) -> float:
        all_rooted_paths = self.all_rooted_paths
        len_idx = np.argsort([len(p) for p in all_rooted_paths])[::-1]

        longest_rooted_path = all_rooted_paths[len_idx[0]]
        furthest_node = longest_rooted_path[-1]

        paths = [np.squeeze(list(nx.all_simple_paths(self.G, source=furthest_node, target=t))) for t in self.term_branches]

        return np.max([d.size for d in paths])

    @property
    def d2h_ratio(self) -> float:
        return self.diameter / self.max_order

    def __repr__(self) -> str:
        return 'graph with {} segments and {} max order'.format(self.num_segments, self.max_order)


def random_tree_gen(scheme: str='c', max_order: int=1, num_segments: int=1, symmetry: float=1.0, d2h_ratio: float=1.5) \
        -> Graph:
    """
    Generate a random tree
    """
    assert scheme in ['c', 's'], 'scheme should be either "c" or "s"'
    gen = random_tree_c if scheme == 'c' else random_tree_s
    G = gen(max_order, num_segments, symmetry, d2h_ratio)

    bifs = []
    for nd in G.nodes:
        children = list(G.successors(nd))
        if children:
            order = len(list(nx.shortest_path(G, source=0, target=nd)))
            bf = {'order': order, 'context': {'p': nd, 'c': children}}
            bifs.append(bf)

    return Graph(bifs, scheme=scheme)

def random_tree_c(max_order: int, num_segments: int, symmetry: float=1.0, d2h_ratio: float=1.5) -> nx.DiGraph:
    """
    construct a random tree with 1 stem and given max levels and number of terminals
    """
    num_segments = np.clip(num_segments, 2 * max_order - 1, np.power(2, max_order) - 1)

    d2h_ratio = np.clip(d2h_ratio, 1.1, 2.0)
    diameter = int(max_order * d2h_ratio)
    diameter = np.clip(diameter, max_order + 1, 2 * max_order - 1)

    avl_nodes = OrderedSet()
    G = nx.DiGraph()
    G.add_node(0)
    avl_nodes.add(0)
    added_nodes_ = 1
    lr_dist = np.nan * np.ones((num_segments, max_order))

    def lr() -> int:
        if np.random.rand() < symmetry / (1 + symmetry):
            return 0
        else:
            return 1

    def _truncate_avail_nodes(added_nodes: list, removed_nodes: list) -> None:
        for an in added_nodes:
            avl_nodes.add(an)
        for rn in removed_nodes:
            avl_nodes.remove(rn)

    def _add_nodes(parent: int, children: list) -> None:

        nonlocal added_nodes_

        G.add_nodes_from(children, m=0)
        G.add_edges_from([(parent, children[0]), (parent, children[1])])

        c1_lr, c2_lr = (lr_dist[parent].copy(), lr_dist[parent].copy())
        idx = np.where(np.isnan(c1_lr))[0][1]
        c1_lr[idx] = 0
        c2_lr[idx] = 1
        lr_dist[children[0]] = c1_lr
        lr_dist[children[1]] = c2_lr

        added_nodes_ += 2

    for i in range(1, max_order):
        cl, cr = (2 * i - 1, 2 * i)
        if i > 1:
            p = 2 * i - 2 if lr() else 2 * i - 3
        else:
            p = 0
        _add_nodes(parent=p, children=[cl, cr])

        if i != max_order - 1:
            _truncate_avail_nodes([cl, cr], [p])
        else:
            _truncate_avail_nodes([], [p])
            highest_order_node = cl

    while num_segments - added_nodes_ >= 2:
        nd_prob = np.array([asym_penalty(node=nd, sym_score=symmetry, lr_dist=lr_dist, max_order=max_order) for nd in avl_nodes])
        while True:
            node_2_add = pick_nodes_wrt_prob(avl_nodes, nd_prob)
            cl, cr = (added_nodes_, added_nodes_ + 1)
            if criteria_max_order(G, node_2_add, max_order - 1) and criteria_diameter(G, highest_order_node, node_2_add, diameter - 1):
                _add_nodes(parent=node_2_add, children=[cl, cr])
                _truncate_avail_nodes([cl, cr], [node_2_add])
                break

    return G


def random_tree_s(max_order: int, num_segments: int, symmetry: float=1.0, d2h_ratio: float=1.5) -> nx.DiGraph:
    """
    construct a random tree with 1 stem and given max levels and number of terminals

    """
    max_order_sub = max_order - 1
    num_segments = np.clip(num_segments, 2 * max_order_sub, 1000.0)

    G = nx.DiGraph()
    G.add_node(0, m=1)
    added_nodes = 1

    while added_nodes < num_segments:
        _max_order = np.random.choice(np.arange(1, max_order_sub + 1))
        _num_segments = np.random.choice(np.arange(2 * max_order_sub - 1, np.power(2, max_order_sub)))

        sub_G = random_tree_c(max_order=_max_order, num_segments=_num_segments, symmetry=symmetry, d2h_ratio=d2h_ratio)
        label_remap = {i: i + added_nodes for i in range(len(sub_G.nodes))}
        sub_G = nx.relabel_nodes(sub_G, label_remap, copy=False)

        G.add_nodes_from(sub_G.nodes(), m=0)
        G.add_edges_from(sub_G.edges())
        G.add_node(added_nodes + len(sub_G.nodes), m=1)
        G.add_edges_from([(added_nodes - 1, added_nodes), (added_nodes - 1, added_nodes + len(sub_G.nodes))])

        added_nodes += len(sub_G.nodes) + 1

    return G


def calc_balance(load_mat: np.ndarray) -> int:
    """
    Calculate the symmetric balance of a tree
    """
    _r = np.nansum(load_mat, axis=0)
    _r_score = np.sum([1 / np.power(2, i) * s for i, s in enumerate(_r)])
    _l = np.nansum(1 - load_mat, axis=0)
    _l_score = np.sum([1 / np.power(2, i) * s for i, s in enumerate(_l)])
    return _r_score - _l_score


def asym_penalty(node: int, sym_score: float=1.0, lr_dist: np.ndarray=None, max_order: int=2) -> int:
    """
    Apply asymmetric penalty to a tree .
    """
    sym_score = 1 - sym_score
    max_asym_score = 0.5
    for i in range(2, max_order + 1):
        max_asym_score += (1 - 1 / np.power(2, i - 2)) * 2

    curr_score = calc_balance(lr_dist) / max_asym_score
    node_score = lr_dist[node].copy()
    change = calc_balance(np.array([node_score])) * 2

    if curr_score < 0:
        sym_score *= -1
    offset = sym_score - curr_score
    penalty = offset * change

    if penalty >= 0:
        penalty_score = np.exp(2 * np.sqrt(penalty))
    else:
        penalty_score = np.exp(-2 * np.sqrt(-penalty))

    return penalty_score

def criteria_max_order(g: nx.DiGraph, target_node: int, max_order: int=2) -> bool:
    od = np.squeeze(list(nx.all_simple_paths(g, source=0, target=target_node)))
    if len(od) > max_order:
        return False
    else:
        return True

def criteria_diameter(g: nx.DiGraph, source_node: int, target_node: int, max_diameter: int=2) -> bool:
    od = np.squeeze(list(nx.all_simple_paths(g, source=source_node, target=target_node)))
    if len(od) > max_diameter:
        return False
    else:
        return True


def pick_nodes_wrt_prob(nodes: List, prob: np.ndarray) -> Any:
    norm_prob = np.cumsum(prob) / np.sum(prob)
    prob_num = np.random.rand()
    idx = np.where(norm_prob > prob_num)[0][0]
    picked_node = nodes[idx]
    return picked_node


def interp_graph(path: str, type_id: int, max_order: int, num_segments: int) -> str:
    """
    Interpolate a graph if the direct max_order and num_segments are not given.
    """
    all_graphs = [os.path.splitext(s)[0] for s in os.listdir(path)]
    avail_types = np.unique([int(s.split('_')[1]) for s in all_graphs])

    if type_id not in avail_types:
        type_id = avail_types[np.argmin(np.abs(avail_types - type_id))]
    type_graphs = [s for s in all_graphs if s.startswith('type_{}'.format(type_id))]

    avail_max_order = np.unique([int(s.split('_')[5]) for s in type_graphs])
    if max_order not in avail_max_order:
        max_order = avail_max_order[np.argmin(np.abs(avail_max_order - max_order))]

    order_graphs = [s for s in type_graphs if s.endswith('max_{}'.format(max_order))]
    avail_num_segments = np.unique([int(s.split('_')[3]) for s in order_graphs])

    if num_segments not in avail_num_segments:
        num_segments = avail_num_segments[np.argmin(np.abs(avail_num_segments - num_segments))]

    return os.path.join(path, 'type_{}_num_{}_max_{}.pkl'.format(type_id, num_segments, max_order))


