from typing import Any
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from typing import Union, Iterable, List
from components.base_classes import Node, Element
from utils.graph import Graph
from utils.utils import *
from utils.params_var import interp, global_params

_default_basis = {'x': np.array([1.0, 0, 0]), 'y': np.array([0.0, 1, 0]), 'z': np.array([0.0, 0, 1])}
_LUT = np.load('rsc/lookup_table.npy')
_numeric_zero = 1e3
gp = global_params()
seed = 123
np.random.seed(seed)


class Neurite(Element):

    def __init__(self, start: Node=None, length: float=100.0, init_vec: Union[np.ndarray, tuple]=(np.pi / 2, np.pi / 2),
                 proj_vec: Union[np.ndarray, tuple]=(np.pi / 2, np.pi / 2), init_r: float=1.0, neurite_id: int=0,
                 seg_type: int=2, step_length: np.ndarray=None, angles: Iterable=None, order: int=0, from_soma: bool=False,
                 contr_factor: float=0.9, basis: dict=None, hist_w: float=0.5, proj_w: float=1, reg_basis: dict=None,
                 from_spline: bool=True) -> None:
        """
        Create a piece of neurite from parameters. The skeleton is constructed using addition of node with given
        step size and direction, which is obtained from change in Euler angles.

        :param start: starting node with position and initial radius
        :param length: total path length (in micron)
        :param init_vec: initial vector specifying direction
        :param proj_vec: the guidance vector for neurite growth
        :param init_r: initial radius
        :param neurite_id: neurite id
        :param seg_type: the branch type, which follows literature rule, e.g. 1=soma, 2=axon, ...
        :param step_length: mean step length of the node increment
        :param angles: list of euler angles between adjacent node.
        :param order: rotation order
        :param from_soma: whether this branch is connected to soma or not
        :param contr_factor: radius contraction factor from start to end. <1 -> contraction, >1 -> expansion
        :param basis: the initial basis for the rotation
        :param hist_w: coefficient of the history of the path, which reduces the randomness of the curvature
        :param proj_w: coefficient of the proj_vec, which alters the direction towards a given direction
        :param reg_basis: registration basis if the outcome needs to be in specific orientation
        :param from_spline: options to reconstruct the neurite from spline control points
        """
        super(Neurite, self).__init__()
        self.id = neurite_id
        self.type = seg_type
        self.from_soma = from_soma
        self.order = order
        self.is_term = False
        self.nodes = [start]
        self.skl = init_skl()
        if len(init_vec) == 2:
            self.init_vec = vec_from_sph(phi=init_vec[0], theta=init_vec[1], order='yzy')
        else:
            self.init_vec = init_vec
        if len(proj_vec) == 2:
            self.init_vec = vec_from_sph(phi=proj_vec[0], theta=proj_vec[1], order='yzy')
        else:
            self.proj_vec = proj_vec
        self.init_r = init_r
        self.max_length = length
        if self.max_length < 0:
            self.max_length = 1.0
        self.step_length = step_length
        self.length = 0
        self.vec_path = []
        self.angles = angles
        self.phi = [a[0] for a in self.angles]
        self.theta = [a[1] for a in self.angles]
        if basis is None:
            if self.id == 0:
                self._basis = _default_basis
            else:
                self._basis = init_basis(main_vec=self.init_vec, main_axis='y')
        else:
            self._basis = basis
        self.inarg_basis = self._basis.copy()
        if from_spline:
            self._basis = reg_basis
        self.reg_basis = reg_basis
        self.basis_hist = []
        self.contr_factor = contr_factor
        self.hist_w = hist_w
        self.proj_w = proj_w


    def from_nodes(self, nodes: list) -> None:
        """
        Reconstruct neurite form list of defined nodes
        """
        self.nodes[0] = nodes[0]
        for n in nodes[1:]:
            self._add_node(n)
        self.vertices = np.vstack((nodes[0].pos, self.vertices))
        self.vertex_types = np.insert(self.vertex_types, 0, nodes[0].type)
        self.radius = np.insert(self.radius, 0, nodes[0].r)

    def get_terminals(self) -> None:
        pass

    def grow(self) -> None:
        """
        Growth function for a neurite with defined parameters
        """
        # define initial setup
        num_nodes = 0
        if self.reg_basis is not None:
            (_p, _t), offset = align_basis(old_basis=self.reg_basis, new_basis=self._basis)
        self.init_vec = vec_from_sph(phi=self.phi[0], theta=self.theta[0], basis=self._basis, order='yzy')
        self._basis = rot_basis(old_basis=self._basis, order='yzy', intrinsic=False, beta=self.phi[0], gamma=self.theta[0])
        growth_vec = self.init_vec * self.step_length[0]

        self.skl.vertices = np.vstack((self.skl.vertices, self.nodes[-1].pos + growth_vec))
        self.basis_hist.append(self._basis)
        num_nodes += 1
        init_length = norm(growth_vec)
        self.length = init_length

        # iterative adding nodes until defined path length
        while self.max_length - self.length > _numeric_zero:

            pos = self.length / self.max_length
            phi = interp(self.phi, self.step_length, pos)
            theta = interp(self.theta, self.step_length, pos)
            new_vec = vec_from_sph(phi=phi, theta=theta, basis=self._basis, order='yzy')

            scale_factor = self.step_length[num_nodes]
            growth_vec = new_vec * scale_factor

            # post-processing of neurite curvature
            hist_vec = hist_avg(self.vec_path)
            proj_vec = self.proj_vec
            proj_w = proj_bias_weight(pos, self.proj_w)

            growth_vec = final_vec(growth_vec, other_vecs=(hist_vec, proj_vec), weights=(self.hist_w, proj_w))

            self.length += norm(growth_vec)
            vertice_pos = self.skl.vertices[-1] + growth_vec
            self.skl.vertices = np.vstack((self.skl.vertices, vertice_pos))
            self.vec_path.append(growth_vec)

            num_nodes += 1

            # update basis
            self._basis = rot_basis(old_basis=self._basis, order='yzy', intrinsic=False, beta=phi, gamma=theta)
            self.basis_hist.append(self._basis)

        if self.reg_basis is not None:
            (_p, _t), offset = align_basis(old_basis=self.reg_basis, new_basis=self.inarg_basis)
            v_ = self.skl.vertices.copy()
            rot_v = np.array([0, 0, 0])
            for i in range(1, v_.shape[0]):
                vec = v_[i] - v_[i - 1]
                temp_vec = axis_rot(self.reg_basis['x'], vec, angle=_p)
                temp_vec = axis_rot(self.reg_basis['y'], temp_vec, angle=_t)
                temp_vec = axis_rot(self.inarg_basis['y'], temp_vec, angle=offset)
                rot_v = np.vstack((rot_v, temp_vec + rot_v[-1]))
            self.skl.vertices = rot_v + self.nodes[-1].pos

        self.skl.vertices = self.skl.vertices[1:]
        if num_nodes == 1:
            self.edges = np.zeros((0, 2), dtype=np.uint8)
        else:
            self.skl.edges = np.vstack([[i, i + 1] for i in range(num_nodes - 1)])
        self.skl.vertex_types = np.ones(num_nodes, dtype=int) * self.type

        radii = np.ones(num_nodes) * self.init_r * np.linspace(start=1.0, stop=self.contr_factor, num=num_nodes)
        radii = np.clip(radii, 0.0, 100.0)
        self.skl.radii = radii

    def attach_neurite(self, neurite, conn: Union[str, int, Node]='first') -> None:
        """
        Attach neurite to existing skeleton at specific node connection.
        """
        if conn == 'first':
            conn_node = neurite.nodes[0]
            pre_num = self.num_nodes
            new_vertices = neurite.vertices
            for n_node in neurite.nodes[1:]:
                n_node.id += pre_num - 2
            new_edges = neurite.edges[1:] + pre_num
            self.vertices = np.vstack((self.vertices, new_vertices))
            self.vertex_types = np.concatenate((self.vertex_types, neurite.vertex_types))
            self.radius = np.concatenate((self.radius, neurite.radius))
            self.edges = np.vstack((self.edges, np.array([conn_node.id, pre_num + 1]), new_edges))
            self.nodes.extend(neurite.nodes[1:])
        else:
            pre_num = self.num_nodes
            new_vertices = neurite.vertices
            for n_node in neurite.nodes:
                n_node.id += pre_num
            self.vertices = np.vstack((self.vertices, new_vertices))
            self.vertex_types = np.concatenate((self.vertex_types, neurite.vertex_types))
            self.radius = np.concatenate((self.radius, neurite.radius))
            new_edges = neurite.edges + pre_num
            if isinstance(conn, int) or isinstance(conn, Node):
                self.edges = np.vstack((self.edges, [conn.id, pre_num], new_edges))
            else:
                self.edges = np.vstack((self.edges, new_edges))
            self.nodes.extend(neurite.nodes)

    def _add_node(self, node: Node) -> None:
        """
        Helper function for adding node to the last node.
        """
        self.radius = np.append(self.radius, node.r)
        self.vertex_types = np.append(self.vertex_types, node.type)
        new_node_id = node.id
        if self.num_nodes > 0:
            self.vertices = np.vstack((self.vertices, node.pos))
        else:
            self.vertices = np.array([node.pos])
        if len(self.edges) > 0 and node.p is not None:
            if len(node.p) == 1:
                self.edges = np.vstack((self.edges, [node.p[0].id, new_node_id]))
            elif isinstance(node.p, list):
                self.edges = [np.vstack((self.edges, [loc, new_node_id])) for loc in node.p]
            else:
                raise ValueError('parent node does not exist.')
        elif len(self.edges) == 0 and node.p[0] == self.nodes[0]:
            self.edges = np.array([0, 1])
        self.nodes.append(node)

    def _delete_node(self, node: Union[Node, int]) -> None:
        """
        Helper function for deleting a specific node.
        """
        if isinstance(node, Node):
            node_pos_id = np.nonzero([n.id == node.id for n in self.nodes])[0][0]
        else:
            node_pos_id = node
            assert node <= len(self.nodes) + 1, 'nodes index exceeding number of nodes'
        self.nodes.pop(node_pos_id)
        node_v_pos = np.argmin(cdist(self.vertices, np.array([node.pos])))
        self.vertices = np.vstack((self.vertices[0:node_v_pos], self.vertices[node_v_pos + 1:]))
        self.radius = np.concatenate((self.radius[0:node_v_pos], self.radius[node_v_pos + 1:]))
        self.vertex_types = np.concatenate((self.vertex_types[0:node_v_pos], self.vertex_types[node_v_pos + 1:]))
        edges_pos = np.where(self.edges == node_pos_id)[0]
        edges_retain = [e for e in range(self.edges.shape[0]) if e not in edges_pos]
        self.edges = self.edges[edges_retain, :]
        self.edges[self.edges > node_pos_id] -= 1

    def get_vec_path(self) -> List:
        """
        Return the vec path.
        """
        assert len(self.nodes) > 1, 'must have at least two nodes'
        path = []
        for i in range(len(self.nodes) - 1):
            path.append(self.nodes[i + 1].pos - self.nodes[i].pos)
        return path

    @staticmethod
    def from_spline(t: np.ndarray, ctrl_pts: Any, k: int, reg_basis: dict, norm_list: list, step_sizes: np.ndarray) -> Any:
        """
        Reconstruct the angles from from spline control points.
        """
        phi_list, theta_list = ctrl_pts
        basis = _default_basis
        new_ctrl_pts = np.array([0, 0, 0])
        for phi, theta, nm in zip(phi_list, theta_list, norm_list):
            new_vec = vec_from_sph(phi=phi, theta=theta, basis=basis, order='yzy')
            new_vec *= nm
            new_ctrl_pts = np.vstack([new_ctrl_pts, new_ctrl_pts[-1] + new_vec])
            basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=phi, gamma=theta)
        new_tck = tuple([t, new_ctrl_pts.T, k])
        step_sizes = np.concatenate([[0], step_sizes])
        new_u = np.cumsum(step_sizes) / np.sum(step_sizes)
        new_pts = splev(new_u, tck=new_tck)
        x_new, y_new, z_new = new_pts
        sampled_vertices = np.array([x_new, y_new, z_new]).T
        basis = reg_basis
        phi_list, theta_list = ([], [])
        for i in range(1, sampled_vertices.shape[0]):
            vec = sampled_vertices[i] - sampled_vertices[i - 1]
            vec = vec / norm(vec)
            phi, theta = get_sph_angles(carte=vec, basis=basis, order='yzy')
            phi_list.append(phi)
            theta_list.append(theta)
            basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=phi, gamma=theta)
        return phi_list, theta_list

    def __str__(self) -> str:
        if self.vertices.shape[0] == len(self.nodes):
            return 'Isolated neurite #{} with {} nodes, and {} edges'.format(self.id, self.vertices.shape[0], self.edges.shape[0])
        else:
            return 'Connected neurite #{} to node # {}, with {} nodes, and {} edges'.format(self.id, self.nodes[0].id, self.skl.vertices.shape[0], self.skl.edges.shape[0])

    def __repr__(self) -> str:
        return self.__str__()


class Bifurcation(Element):

    def __init__(self, tilt: float, ratio: float, ampl: float, p_vec: np.ndarray, roll: float=0.0, p_r: float=1.0,
                 aux_vec: np.ndarray=np.array([1.0, 0.0, 0.0]), order: int=1) -> None:
        """
        The class of creating a bifurcation.
        :param tilt: the plane constituted by two child vectors vs. the previous vector
        :param ratio: split ratio of two child vectors. always >=1
        :param ampl: the total span angles of the two child vectors
        :param roll: the roll angle of the two child vectors around the axis p_vec
        :param p_vec: parent vector
        :param p_r: parent segment radius
        :param aux_vec: the previous vector of the parent vector
        """
        super(Bifurcation, self).__init__()
        self.tilt = abs(tilt)
        self.ratio = ratio
        self.ampl = ampl
        self.roll = roll
        self.p_vec = np.array(p_vec) / norm(p_vec)
        aux_vec = residual(p_vec, aux_vec)
        if norm(aux_vec) < 0.01:
            aux_vec = np.array([1.0, 0, 0])
        if norm(np.cross(p_vec, aux_vec)) < 0.01:
            aux_vec = np.array([0.0, 0.0, 1.0])
        self.aux_vec = aux_vec / norm(aux_vec)
        self.norm_p = np.cross(self.aux_vec, self.p_vec)
        self.p_basis = {'y': self.p_vec, 'x': self.aux_vec, 'z': self.norm_p}
        self.p_r = p_r
        self.order = order

    def make_bif(self) -> dict:
        """
        Make the geometry of the bifurcation following the zyz order with the intrinsic basis formed by
        the parent vector and its parent vector. x is the p_vec, y is the aux_vec (residual of pp_vec
        onto p_vec), z is p_vec x aux_vec.
        """
        tilt = -self.tilt if np.random.rand() > 1.0 else self.tilt
        c_basis = rot_basis(old_basis=self.p_basis, alpha=0.0, beta=tilt, gamma=self.roll, intrinsic=False, order='yzy')
        angle = self.ampl / 2
        z = c_basis['z']
        y = c_basis['y']
        asym = self.calc_asym(lut=_LUT)
        angle_1 = -angle + asym
        angle_2 = angle + asym
        if np.abs(angle_1) > np.abs(angle_2):
            angle_1, angle_2 = (angle_2, angle_1)
        c_vec_1 = axis_rot(z, y, angle_1)
        c_vec_2 = axis_rot(z, y, angle_2)
        proj_degree = np.clip(gp.expand_coef, 0.1, 2)
        proj_vec_1 = axis_rot(z, y, angle_1 * proj_degree)
        proj_vec_2 = axis_rot(z, y, angle_2 * proj_degree)
        c_basis_1 = rot_basis(old_basis=c_basis, alpha=0.0, beta=angle_1, gamma=0, order='zyz', intrinsic=False)
        c_basis_2 = rot_basis(old_basis=c_basis, alpha=0.0, beta=angle_2, gamma=0, order='zyz', intrinsic=False)
        r_1, r_2 = rall_rule(r_p=self.p_r, split=self.ratio, n=gp.rall, sp_decay_coef=0.1)
        if r_1 < r_2:
            r_1, r_2 = (r_2, r_1)
        return {'vec': [c_vec_1, c_vec_2], 'r': [r_1, r_2], 'proj_vec': [proj_vec_1, proj_vec_2], 'basis': [c_basis_1, c_basis_2]}

    def calc_asym(self, lut: Any=_LUT) -> Any:
        """
        Calc asymmetry of the two child vectors. 1 means even split. To speed up, the answer is interpolated from
        pre-computed lookup tables.
        """
        ampl_array = np.linspace(0, np.pi, 201)
        tilt_array = np.linspace(0, np.pi / 2, 101)
        asym_array = np.linspace(-np.pi / 2, np.pi / 2, 201)
        ampl_index = np.argmin(np.abs(ampl_array - abs(self.ampl)))
        tilt_index = np.argmin(np.abs(tilt_array - abs(self.tilt)))
        ratio_array = lut[ampl_index, tilt_index, :]
        diff = np.array([dd * ddd for dd, ddd in zip(ratio_array[:-1] - self.ratio, ratio_array[1:] - self.ratio)])
        idx_cand = np.argwhere(diff < 0)
        if len(idx_cand) == 1:
            idx = idx_cand[0]
        elif len(idx_cand) == 2:
            asym_cand = asym_array[idx_cand]
            idx = idx_cand[np.argmin(np.abs(np.abs(asym_cand) - self.ampl / 2))]
        else:
            idx = np.argmin(np.abs(ratio_array - self.ratio))
        idx = np.squeeze(idx)
        solution = asym_array[idx]
        return solution


class Axon(Element):
    """
    Axon is the collections of connected neurite, along with all the bifurcations within.
    """
    def __init__(self, comp: dict, **kwargs) -> None:
        super(Axon, self).__init__()
        self.graph = {}
        self.skl = init_skl()
        if 'id' in kwargs.keys():
            self.id = kwargs['id']
        if 'type_id' in kwargs.keys():
            self.type_id = kwargs['type_id']
        self.max_order = 1
        self.num_term = 1
        self.grown_neurite = {}
        self.neurite_list = comp['segments']
        self.bif_list = comp['bifs']
        self.graph = Graph(self.bif_list)
        self.steps = self.graph.trs_steps()

    def grow(self) -> None:
        """
        Grow neurite and bifurcations with BFS order. The order of branch is the path distance to the root.
        """
        for i in range(1, self.graph.max_order + 1):
            deg_branches = list(self.steps[i])
            for n_id in deg_branches:
                if isinstance(list(self.neurite_list.keys())[0], str):
                    neurite = self.neurite_list[str(n_id)]
                else:
                    neurite = self.neurite_list[n_id]
                if n_id == 0:
                    start_pt = Node(pos=np.array([0, 0, 0]))
                    start_id = -1
                    from_soma = True
                    init_r = neurite['init_r']
                    init_vec = neurite['init_vec']
                    proj_vec = init_vec
                    basis = None
                else:
                    start_pt = Node(pos=self.skl.vertices[self.graph.conn_pt[n_id]])
                    start_id = self.graph.conn_pt[n_id]
                    from_soma = False
                    p_seg_id = self.graph.get_parent(n_id)
                    bch_info = [bch for bch in self.bif_list if int(bch['context']['p']) == int(p_seg_id)][0]
                    tilt = bch_info['tilt']
                    ampl = bch_info['ampl']
                    roll = bch_info['roll']
                    ratio = bch_info['ratio']
                    if neurite['from_spline'] and 'reverse' in bch_info.keys():
                        reverse = bch_info['reverse']
                    else:
                        reverse = False
                    if len(self.grown_neurite[p_seg_id].skl.vertices) > 1:
                        p_vec = self.grown_neurite[p_seg_id].skl.vertices[-1] - self.grown_neurite[p_seg_id].skl.vertices[-2]
                    else:
                        pp_seg_id = self.graph.get_parent(p_seg_id)
                        p_vec = self.grown_neurite[p_seg_id].skl.vertices[-1] - self.grown_neurite[pp_seg_id].skl.vertices[-1]
                    if len(self.grown_neurite[p_seg_id].skl.vertices) > 2:
                        pp_vec = self.grown_neurite[p_seg_id].skl.vertices[-2] - self.grown_neurite[p_seg_id].skl.vertices[-3]
                    else:
                        pp_vec = np.array([0.0, 1.0, 0.0])
                    p_r = self.grown_neurite[p_seg_id].skl.radii[-1]
                    branch_info = Bifurcation(tilt=tilt, ratio=ratio, ampl=ampl, roll=roll, p_vec=p_vec, aux_vec=pp_vec, p_r=p_r).make_bif()
                    bch_index = np.where(np.array(bch_info['context']['c']) == int(n_id))[0][0]
                    if reverse:
                        bch_index = 1 - bch_index
                    init_vec = branch_info['vec'][bch_index]
                    init_r = branch_info['r'][bch_index]
                    proj_vec = branch_info['proj_vec'][bch_index]
                    basis = branch_info['basis'][bch_index]
                n = Neurite(start=start_pt, init_vec=init_vec, length=neurite['length'], step_length=neurite['step_length'], angles=neurite['rel_angles'], neurite_id=n_id, order=i, seg_type=self.type_id, from_soma=from_soma, init_r=init_r, proj_vec=proj_vec, basis=basis, proj_w=neurite['proj_w'], contr_factor=neurite['contr_factor'], hist_w=neurite['hist_w'], reg_basis=neurite['reg_basis'], from_spline=neurite['from_spline'])
                n.grow()
                self.add_skl(new_skl=n.skl, conn_idx=start_id)
                if len(self.graph.get_children(idx=n_id)) > 0:
                    self.graph.update_conn(self.graph.get_children(idx=n_id), conn=self.skl.vertices.shape[0] - 1)
                self.grown_neurite.update({int(n_id): n})

    def __str__(self) -> str:
        return 'Axon #{} with {} nodes and {} neurite'.format(self.id, self.skl.vertices.shape[0], len(self.grown_neurite.keys()))


def hist_avg(vec_path: np.ndarray, num: int=3, decay_const: float=2) -> np.ndarray:
    """
    Return the history averaged neurite with given coefficients.
    """
    if isinstance(vec_path, list):
        if len(vec_path) == 0:
            vec_path = np.zeros((0, 3))
        elif len(vec_path) > 1:
            vec_path = np.vstack(vec_path)
    pad = np.zeros((num, 3))
    vec_path = np.vstack((pad, vec_path))
    vecs = vec_path[-num:]
    vec_len = norm(vecs, axis=1)
    vec_len_cum = np.cumsum(vec_len)
    total_len = np.sum(vec_len) + 1e-05
    weight = np.exp(-decay_const * (vec_len_cum / total_len))
    hist_vec = np.sum([v * w for v, w in zip(vecs, weight)], axis=0)
    return hist_vec


def proj_bias_weight(pos: float, base_weight: float) -> float:
    """
    Return the proj vector averaged neurite with given coefficients.
    """
    if pos < 0.1:
        return base_weight * (2 * np.exp(-10 * pos) - 1)
    elif pos > 0.9:
        return base_weight * (2 * np.exp(-10 * (1 - pos)) - 1)
    else:
        return base_weight


def final_vec(ori_vec: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computer the final direction vector after post-processing
    """
    if 'other_vecs' in kwargs.keys() and 'weights' in kwargs.keys():
        other_vecs = kwargs['other_vecs']
        weight = kwargs['weights']
        assert len(other_vecs) == len(weight), 'weight and vectors much match in number'
        ori_norm = norm(ori_vec)
        purt_vec = np.sum([v * w / (norm(v) + 0.001) for v, w in zip(other_vecs, weight)], axis=0)
        new_vec = ori_vec / ori_norm + purt_vec
        new_vec = new_vec * ori_norm / norm(new_vec)
        return new_vec

    else:
        return ori_vec


if __name__ == '__main__':
    pass
