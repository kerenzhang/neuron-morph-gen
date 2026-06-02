import numpy as np
from numpy.linalg import norm
from cloudvolume import Skeleton
from typing import Union, List
from scipy.spatial.distance import cdist
from utils.utils import init_skl

class Node:

    def __init__(self, pos: Union[list, np.ndarray]=None, r: float=0, type: float=0, **kwargs) -> None:
        """
        Construct a node with position and radius.
        """
        self.id = -1
        self.pos = np.array(pos, dtype=float)
        self.r = r
        self.type = type
        self.p = []
        self.c = []
        if kwargs:
            for key, value in kwargs.items():
                self.__setattr__(key, value)

    @property
    def connectivity(self) -> int:
        """
        Return the connectivity of the node (how many nodes are connected to this node)
        """
        return max(len(self.p), len(self.c))

    @property
    def curvature(self) -> List:
        """Return the curvature."""
        if self.connectivity == 1:
            vec_p = self.pos - self.p[0].pos
            vec_c = self.c[0].pos - self.pos
            return np.dot(vec_p, vec_c) / (norm(vec_c) * norm(vec_p))
        else:
            neighbors = self.p if len(self.p) > len(self.c) else self.c
            curvature = []
            for nbr in neighbors:
                vec_p = self.pos - nbr.pos
                vec_c = nbr.pos - self.pos
                curvature.append(np.dot(vec_p, vec_c) / (norm(vec_c) * norm(vec_p)))
            return curvature

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            if np.sum(self.pos - other.pos) < 0.001:
                return True
        return False

    def __str__(self) -> str:
        return 'Node# {} at {}, r={}, type={}'.format(self.id, list(self.pos), self.r, self.type)

    def __repr__(self) -> str:
        return self.__str__()


class Element(Skeleton):

    def __init__(self) -> None:
        """
        Base class for subcomponents of a neuron, which include neurite, axons, bifurcations, and soma.
        """
        super(Element, self).__init__()
        self.skl = init_skl()
        self.start_pt = np.array([0, 0, 0])
        self.nodes = []
        self.id = 0

    def _add_node(self, node: Node) -> None:
        """
        Add new node to existing node.
        """
        self.skl.radius = np.append(self.skl.radius, node.r)
        self.skl.vertex_types = np.append(self.skl.vertex_types, node.type)
        new_node_id = node.id

        if self.num_nodes > 0:
            self.skl.vertices = np.vstack((self.skl.vertices, node.pos))
        else:
            self.skl.vertices = np.array([node.pos])

        if len(self.edges) > 0 and node.p is not None:
            if len(node.p) == 1:
                self.skl.edges = np.vstack((self.skl.edges, [node.p[0], new_node_id]))
            elif isinstance(node.p, list):
                self.skl.edges = [np.vstack((self.skl.edges, [loc, new_node_id])) for loc in node.p]
            else:
                raise ValueError('parent node does not exist')
        elif len(self.edges) == 0 and node.p[0] == 0:
            self.skl.edges = np.array([0, 1])

        self.nodes.append(node)

    def add_skl(self, new_skl: Skeleton, offset: np.ndarray=np.array([0, 0, 0]), conn_idx: int=-1) -> None:
        """
        Add skeleton to existing skeletons. The offset is the position of the node to be connected.
        """
        num_nodes = self.skl.vertices.shape[0]
        self.skl.vertices = np.vstack((self.skl.vertices, new_skl.vertices + offset))
        self.skl.vertex_types = np.concatenate((self.skl.vertex_types, new_skl.vertex_types))
        self.skl.radius = np.concatenate((self.skl.radius, new_skl.radius))

        if conn_idx >= 0:
            self.skl.edges = np.vstack((self.skl.edges, [conn_idx, num_nodes], new_skl.edges + num_nodes))
        else:
            self.skl.edges = np.vstack((self.skl.edges, new_skl.edges + num_nodes))

    @staticmethod
    def find_node_idx(vertices: np.ndarray, node_pos: np.ndarray) -> np.int64:
        """
        Find node index based on position. The closet point would be chosen
        """
        dist_mat = cdist(vertices, np.array([node_pos]))
        idx = np.argmin(dist_mat)

        return idx

    @staticmethod
    def find_node_neighbors(skl: Skeleton, node_idx: int, return_pos: bool=True) -> int:
        """Find node neighbors."""
        conn = np.argwhere(skl.edges == node_idx)
        conn[:, 1] = 1 - conn[:, 1]
        neigh_idx = skl.edges[conn[:, 0], conn[:, 1]]
        if return_pos:
            return skl.vertices[neigh_idx]
        else:
            return neigh_idx

    @staticmethod
    def is_in_seg(skl: Skeleton, node: Union[int, np.ndarray]) -> bool:
        """
        Check if node is in a branch segment. The threshold is 1e-5.
        """
        if isinstance(node, np.ndarray):
            dist_mat = cdist(skl.vertices, np.array([node]))
            dist_list = norm(dist_mat, axis=1)
            if np.min(dist_list) < 1e-05:
                return True
        elif isinstance(node, int):
            if node in skl.edges:
                return True
        return False

    @staticmethod
    def delete_node(skl: Skeleton, node: np.ndarray) -> Skeleton:
        """
        Delete node and return the new skeleton.
        """
        node_idx = Element.find_node_idx(skl.vertices, node_pos=node)
        edges = skl.edges.copy()
        conn = np.argwhere(skl.edges == node_idx)[:, 0]
        edges = np.delete(edges, conn, axis=0)

        vertices = skl.vertices.copy()
        vertices = np.delete(vertices, node_idx, axis=0)

        vertex_type = skl.vertex_types.copy()
        vertex_type = np.delete(vertex_type, node_idx)

        radius = skl.radius.copy()
        radius = np.delete(radius, node_idx)

        edges[edges > node_idx] -= 1
        new_skl = Skeleton(vertices=vertices, radii=radius, vertex_types=vertex_type, edges=edges)

        return new_skl

    @property
    def num_nodes(self) -> int:
        return len(self.vertices)


