from typing import Union, List, Tuple
from cloudvolume import Skeleton
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
import scipy
import json
from utils.graph import Graph
from utils.utils import *

_vec_order = 'yzy'

class Axon_info:
    """
    Extract on the axon level
    """
    def __init__(self, type_id: int, graph: Union[Graph, List], init_vec: np.ndarray, init_r: float) -> None:
        """
        Default is BFS and centrifugal order
        """
        self.type_id = type_id
        if not isinstance(graph, Graph):
            self.graph = Graph(graph)
        else:
            self.graph = graph
        self.max_order = self.graph.max_order
        self.num_segments = self.graph.num_segments
        self.init_vec = init_vec
        self.info = {'init_vec': self.init_vec, 'graph': self.graph, 'max_order': self.max_order,
                     'type_id': self.type_id, 'num_segments': self.num_segments, 'init_r': init_r}


class Neurite_seg_info:
    """
    Contains all parameters to be extracted of a single neurite.
    """
    def __init__(self, skl: Skeleton, extract_exact: bool=False, parent_basis: dict=None, **kwargs) -> None:
        self.skl = skl
        self.info = ['length', 'step_size', 'init_vec', 'term_vec', 'avg_vec', 'phi', 'theta', 'type_id', 'order']
        if extract_exact:
            self.info.append('spl_ctrl_pts')
        self.length = 0
        self.order = 0
        self.angles = []
        self.step_size = []
        self.stem = False
        self.init_vec = []
        self.avg_vec = []
        self.term_vec = []
        self.abs_angles = []
        self.phi = []
        self.theta = []
        self.type_id = int(skl.vertex_types[-1])
        self.spl_ctrl_pts = []
        self.extract_exact = extract_exact
        if parent_basis is None:
            init_vec = self.get_init_vec(return_vec=True)
            self._basis = init_basis(main_vec=init_vec, main_axis='y')
        else:
            self._basis = parent_basis
        self.reg_basis = self._basis.copy()
        self.kwargs = kwargs

    def get_abs_angles(self) -> None:
        v = self.skl.vertices
        for i in range(1, v.shape[0]):
            g_vec = v[i] - v[i - 1]
            phi, theta = get_sph_angles(g_vec)
            self.abs_angles.append((float(phi), float(theta)))

    def get_phi(self) -> None:
        v = self.skl.vertices
        for i in range(1, v.shape[0]):
            g_vec = v[i] - v[i - 1]
            phi, theta = get_sph_angles(carte=g_vec, basis=self._basis, order=_vec_order)
            self.phi.append(float(phi))
            self.theta.append(float(theta))
            self._update_basis(phi, theta)

    def get_theta(self) -> None:
        pass

    def get_order(self) -> int:
        if 'order' in self.kwargs.keys():
            self.order = self.kwargs['order']
            return self.order
        else:
            return -1

    def get_type_id(self) -> int:
        return self.type_id

    def get_init_vec(self, return_vec: bool=True) -> np.ndarray:
        init_vec = self.skl.vertices[1] - self.skl.vertices[0]
        if return_vec:
            self.init_vec = [float(v) for v in init_vec]
        else:
            self.init_vec = get_sph_angles(init_vec, order='yzy')
        return init_vec

    def get_avg_vec(self) -> np.ndarray:
        v = self.skl.vertices
        avg = np.zeros((0, 3), dtype=float)
        for i in range(1, v.shape[0]):
            p_vec = v[i] - v[i - 1]
            avg = np.vstack((avg, p_vec))
        self.avg_vec = [float(v) for v in np.mean(avg, axis=0)]
        return np.mean(avg, axis=0)

    def get_term_vec(self) -> List:
        term_vec = self.skl.vertices[-1] - self.skl.vertices[-2]
        self.term_vec = [float(v) for v in term_vec]
        return term_vec

    def get_length(self) -> None:
        v = self.skl.vertices
        for i in range(1, v.shape[0]):
            p_vec = v[i] - v[i - 1]
            self.length += float(norm(p_vec))

    def get_step_size(self) -> None:
        v = self.skl.vertices
        for i in range(1, v.shape[0]):
            p_vec = v[i] - v[i - 1]
            self.step_size.append(float(norm(p_vec)))

    def get_spl_ctrl_pts(self) -> None:
        self.spl_ctrl_pts = get_ctrl_pts(self.skl, self.reg_basis)

    def _update_basis(self, phi: float, theta: float) -> None:
        self._orth_basis = rot_basis(old_basis=self._basis, order='yzy', intrinsic=False, alpha=0, beta=phi, gamma=theta)

    def get_info(self) -> dict:
        neurite_info = {}
        for info in self.info:
            getattr(self, 'get_' + info)()
            neurite_info.update({info: getattr(self, info)})
        return neurite_info


class Bifurcation_info:
    """
    Contains all parameters to be extracted of a bifurcation.
    """
    def __init__(self, p_seg: Skeleton, c_segs: list, order: int, context: dict, type_id: int=2, extract_exact: bool=False) -> None:
        """
        Extract the geometry of the bifurcation and the context info.
        p_seg: parent segment
        c_segs: list of child segments
        order: order of the bifurcation
        context: context info about the id of parent and children
        type_id: the type_id of the host neurite
        """
        self.p_seg = p_seg
        self.c_segs = c_segs
        self.rall_ratio = 0
        self.tilt = 0.0
        self.ratio = 0.0
        self.ampl = 0.0
        self.order = order
        self.roll = 0.0
        self.type_id = type_id
        self.reverse = False
        self.extract_exact = extract_exact

        p_vec = p_seg.vertices[-1] - p_seg.vertices[-2]
        self.p_vec = p_vec / norm(p_vec)
        bif_1_vec = self.c_segs[0].vertices[1] - self.c_segs[0].vertices[0]
        self.bif_1_vec = bif_1_vec / norm(bif_1_vec)
        bif_2_vec = self.c_segs[1].vertices[1] - self.c_segs[1].vertices[0]
        self.bif_2_vec = bif_2_vec / norm(bif_2_vec)
        norm_bif = np.cross(self.bif_1_vec, self.bif_2_vec)
        self.norm_bif = norm_bif / norm(norm_bif)
        self.mid_bif = (self.bif_1_vec + self.bif_2_vec) / 2
        self.mid_bif = self.mid_bif / norm(self.mid_bif)
        c_aux_vec = np.cross(self.mid_bif, self.norm_bif)
        self.c_basis = {'y': self.mid_bif, 'x': c_aux_vec, 'z': self.norm_bif}
        nodes_remaining = p_seg.vertices.shape[0] - 2
        aux_vec = np.array([1, 0, 0])
        while nodes_remaining > 1:
            pp_vec = p_seg.vertices[nodes_remaining] - p_seg.vertices[nodes_remaining - 1]
            pp_vec = pp_vec / norm(pp_vec)
            aux_vec = residual(p_vec, pp_vec)
            if norm(aux_vec) > 0.01:
                break
            nodes_remaining -= 1
        if norm(aux_vec) < 0.01:
            aux_vec = np.array([1.0, 0.0, 0.0])
        if norm(np.cross(p_vec, aux_vec)) < 0.01:
            aux_vec = np.array([0.0, 0.0, 1.0])
        self.aux_vec = aux_vec / norm(aux_vec)
        self.norm_p = np.cross(self.aux_vec, self.p_vec)
        self.norm_p = self.norm_p / norm(self.norm_p)
        self.p_basis = {'y': self.p_vec, 'x': self.aux_vec, 'z': self.norm_p}
        self.context = context
        self.info = ['ampl', 'ratio', 'tilt', 'roll', 'order', 'context', 'type_id']
        if self.extract_exact:
            self.info.append('reverse')

    def get_roll(self) -> None:
        """
        get the roll of the plane of 2 bif children w.r.t the orthogonal basis spanned by the parent vector and its
        parent vector. The default order is zyz, rotating a z vector. roll is equivalent to the phi (relative to the
        aux_vec x p_vec).
        """
        pass

    def get_order(self) -> None:
        pass

    def get_context(self) -> None:
        pass

    def get_type_id(self) -> None:
        pass

    def get_ampl(self) -> float:
        """
        get the amplitude of angle between two children vector
        """
        amplitude = get_angle(self.bif_1_vec, self.bif_2_vec)
        self.ampl = float(amplitude)
        return amplitude

    def get_ratio(self) -> float:
        """
        get the ratio of the two children w.r.t. the projection of parent vector onto the plane, which is not to
        be confused with the continuation of the parent vector. If the ratio is negative, the two children vectors are
        on the same side

        """
        angle_1 = get_angle(self.p_vec, self.bif_1_vec)
        angle_2 = get_angle(self.p_vec, self.bif_2_vec)
        ratio = angle_1 / (angle_2 + 0.001)
        self.ratio = float(ratio)
        if self.extract_exact:
            self.reverse = True if ratio < 1 else False
        return ratio

    def get_tilt(self) -> Tuple:
        phi, theta = get_sph_angles(carte=self.c_basis['y'], basis=self.p_basis, order='yzy')
        fake_c = rot_basis(old_basis=self.p_basis, intrinsic=False, alpha=0, beta=self.tilt, gamma=self.roll, order='yzy')
        z_align_1 = np.abs(np.dot(self.c_basis['z'], fake_c['z']))
        z_align_2 = np.abs(np.dot(self.c_basis['x'], fake_c['z']))
        if z_align_2 > z_align_1:
            self.p_basis['x'], self.p_basis['z'] = (self.p_basis['z'], self.p_basis['x'])
            phi, theta = get_sph_angles(carte=self.c_basis['y'], basis=self.p_basis, order='yzy')
        self.tilt = float(phi)
        self.roll = float(theta)
        return phi, theta

    def get_reverse(self) -> None:
        pass

    def get_info(self) -> dict:
        bif_info = {}
        for info in self.info:
            getattr(self, 'get_' + info)()
            bif_info.update({info: getattr(self, info)})
        return bif_info

    def distribute_basis(self) -> List[dict]:
        """
        Distribute basis to children branches.
        """
        new_y_1 = axis_rot(self.c_basis['z'], self.c_basis['y'], -self.ampl / 2)
        new_x_1 = axis_rot(self.c_basis['z'], self.c_basis['x'], -self.ampl / 2)
        new_y_2 = axis_rot(self.c_basis['z'], self.c_basis['y'], self.ampl / 2)
        new_x_2 = axis_rot(self.c_basis['z'], self.c_basis['x'], self.ampl / 2)
        return [{'x': new_x_1, 'y': new_y_1, 'z': self.c_basis['z']}, {'x': new_x_2, 'y': new_y_2, 'z': self.c_basis['z']}]


def get_seg_dict(skl: Skeleton, exclude_soma: bool=False, node_type: int=1) -> dict:
    seg_paths = [np.array(path) for path in skl.interjoint_paths(return_indices=True)]
    if not exclude_soma:
        return dict([(i, seg) for i, seg in enumerate(seg_paths)])
    else:
        seg_dict = {}
        for i, seg in enumerate(seg_paths):
            if np.sum(skl.vertex_types[seg] == node_type) <= 1:
                seg_dict.update({i: seg})
        return seg_dict


def find_seg(node: int, skl: Skeleton, exclude: int=None, exclude_soma: bool=True) -> List:
    """
    Get the node's belonging of seg paths.
    """
    path_list = []
    seg_dict = get_seg_dict(skl, exclude_soma=exclude_soma)
    for i, seg in seg_dict.items():
        if node in seg and i != exclude:
            path_list.append(i)
    return path_list


def get_stem(skl: Skeleton, soma_type: int=1) -> dict:
    """
    Get the stem neurite fragment.
    """
    stem_nodes = []
    for e in skl.edges:
        node_types = np.array([skl.vertex_types[n] for n in e])
        if np.sum(node_types == soma_type) == 1:
            stem_nodes.append([n for n in e if skl.vertex_types[n] != soma_type][0])
    stems = {}
    for i, stem_n in enumerate(stem_nodes):
        for k, seg in get_seg_dict(skl).items():
            if stem_n in seg:
                vertices = skl.vertices[seg, :]
                radii = skl.radii[seg]
                vertex_types = skl.vertex_types[seg]
                edges = np.vstack([[j, j + 1] for j in range(0, len(seg) - 1)])
                if skl.vertex_types[seg[-1]] == soma_type:
                    seg = seg[::-1]
                    vertices = np.flip(vertices, axis=0)
                    radii = radii[::-1]
                branch_node = seg[-1]
                children = []
                if get_conn(branch_node, skl.edges) > 2:
                    children = find_seg(branch_node, skl, exclude=k)
                stems.update({k: {'skl': Skeleton(vertices=vertices, radii=radii, edges=edges, vertex_types=vertex_types), 'order': 1, 'terminus': [], 'branch_node': branch_node, 'children': children}})
    return stems


def get_seg(skl: Skeleton, seg_dict: dict, index: int, parent_seg_index: int) -> dict:
    """
    Get the stem neurite fragment.
    """
    seg_nodes_idx = seg_dict[index]
    vertices = skl.vertices[seg_nodes_idx, :]
    radii = skl.radii[seg_nodes_idx]
    vertex_types = skl.vertex_types[seg_nodes_idx]
    edges = np.vstack([[i, i + 1] for i in range(len(seg_nodes_idx) - 1)])
    if get_conn(seg_nodes_idx[-1], skl.edges) > 2:
        if seg_nodes_idx[-1] in seg_dict[parent_seg_index]:
            seg_nodes_idx = seg_nodes_idx[::-1]
            vertices = np.flip(vertices, axis=0)
            radii = radii[::-1]
        children = find_seg(seg_nodes_idx[-1], skl, exclude=index)
        terminus = []
        branch_node = seg_nodes_idx[-1]
    else:
        children = []
        terminus = seg_nodes_idx[-1]
        branch_node = None
    new_skl = Skeleton(vertices=vertices, edges=edges, vertex_types=vertex_types, radii=radii)
    seg = {'skl': new_skl, 'order': 1, 'terminus': terminus, 'branch_node': branch_node, 'children': children}
    return seg

def add_seg(skl: Skeleton, seg: Skeleton, branch_pos_idx: int=-1) -> Skeleton:
    pre_num = skl.vertices.shape[0]
    if pre_num > 1:
        edges = np.vstack((skl.edges, [branch_pos_idx, pre_num], seg.edges[1:] + pre_num - 1))
    else:
        edges = np.vstack((skl.edges, seg.edges[1:] - 1))
    vertices = np.vstack((skl.vertices, seg.vertices[1:]))
    radii = np.concatenate((skl.radii, seg.radii[1:]))
    vertex_types = np.concatenate((skl.vertex_types, seg.vertex_types[1:]))
    return Skeleton(vertices=vertices, edges=edges, radii=radii, vertex_types=vertex_types)


def remove_soma(skl: Skeleton, soma_type: int=1) -> Skeleton:
    skl = skl.clone()
    soma_nodes = np.where(skl.vertex_types == soma_type)[0]

    for i, node in enumerate(soma_nodes):
        node = node - i
        node_v_pos = np.argmin(cdist(skl.vertices, np.array([skl.vertices[node]])))
        skl.vertices = np.vstack((skl.vertices[0:node_v_pos], skl.vertices[node_v_pos + 1:]))
        skl.radius = np.concatenate((skl.radius[0:node_v_pos], skl.radius[node_v_pos + 1:]))
        skl.vertex_types = np.concatenate((skl.vertex_types[0:node_v_pos], skl.vertex_types[node_v_pos + 1:]))
        edges_pos = np.where(skl.edges == node_v_pos)[0]
        edges_retain = [e for e in range(skl.edges.shape[0]) if e not in edges_pos]
        skl.edges = skl.edges[edges_retain, :]
        skl.edges[skl.edges > node_v_pos] -= 1
    return Skeleton(vertices=skl.vertices, edges=skl.edges, vertex_types=skl.vertex_types, radii=skl.radii)


def contain_node(skl: Skeleton, vertex: np.ndarray, return_index: bool=True) -> Union[Tuple, bool]:
    dist_matrix = cdist(skl.vertices, np.array([vertex]))
    contains = False

    if np.min(dist_matrix) < 1e-4:
        contains = True
    if return_index:
        index = np.argmin(dist_matrix)
        return contains, index
    else:
        return contains


def detach_soma(skl: Skeleton, soma_type: int=1) -> Tuple:
    """
    construct a separate skl for soma for later use. If there is no soma in the swc, create
    one with center sphere.
    """
    soma_nodes = np.where(skl.vertex_types == soma_type)[0]
    if len(soma_nodes) <= 1:
        if len(soma_nodes) == 1:
            soma_vertices = skl.vertices[soma_nodes, :]
            soma_radii = skl.radii[soma_nodes]
        else:
            soma_vertices = np.array([[0, 0, 0]])
            soma_radii = np.array([2])
        soma_vertex_types = np.array([soma_type])
        soma_edges = np.zeros((0, 2), dtype=int)
    else:
        soma_vertices = np.vstack(skl.vertices[soma_nodes, :])
        soma_vertex_types = skl.vertex_types[soma_nodes]
        soma_radii = skl.radii[soma_nodes]
        soma_edges = np.stack([[i, i + 1] for i in range(len(soma_nodes) - 1)])

    soma_skl = Skeleton(vertices=soma_vertices, vertex_types=soma_vertex_types, radii=soma_radii, edges=soma_edges)
    stem_nodes = []
    aux_soma_nodes = []
    for e in skl.edges:
        node_types = np.array([skl.vertex_types[n] for n in e])
        if np.sum(node_types == soma_type) == 1:
            stem_nodes.append([n for n in e if skl.vertex_types[n] != soma_type][0])
            aux_soma_nodes.append([n for n in e if skl.vertex_types[n] == soma_type][0])

    skl_no_soma = remove_soma(skl, soma_type=soma_type)
    neurite_comp = skl_no_soma.components()

    skl_dict = {}
    for i, comp in enumerate(neurite_comp):
        for aux_soma, stem in zip(aux_soma_nodes, stem_nodes):
            contain, index = contain_node(comp, skl.vertices[stem])
            if contain:
                if np.sum(comp.vertices[0] - skl.vertices[stem]) > 1e-05:
                    comp.vertices = np.flip(comp.vertices, axis=0)
                    comp.radii = np.flip(comp.radii, axis=0)
                    comp.edges = np.vstack(([0, comp.vertices.shape[0] - index], comp.edges + 1))
                else:
                    comp.edges = np.vstack(([0, index + 1], comp.edges + 1))
                comp.vertices = np.vstack((skl.vertices[aux_soma], comp.vertices))
                comp.vertex_types = np.concatenate(([1], comp.vertex_types))
                comp.radii = np.concatenate(([skl.radii[aux_soma]], comp.radii))
                skl_dict.update({i: comp})

    return skl_dict, soma_skl


def extract_axon(skl: Skeleton, soma_type: int=1, extract_exact: bool=False) -> dict:
    """
    extract one single complete axon stemming from soma and contains all its branches and bifurcations
    """
    segs_trsv = set()
    seg_dict = get_seg_dict(skl, exclude_soma=True)
    untrsv_seg = set(seg_dict.keys())
    stem_segs = get_stem(skl=skl, soma_type=soma_type)
    axon_type_id = skl.vertex_types[-1]
    new_skl = init_skl()
    neurite = {}
    bifurcation = []
    basis = {}
    for i, stem in stem_segs.items():
        order = 1
        segs_trsv.add(i)
        untrsv_seg.remove(i)
        new_skl = add_seg(new_skl, stem['skl'])
        init_r = new_skl.radii[0]
        stem_info = Neurite_seg_info(stem['skl'], order=order, extract_exact=extract_exact, parent_basis=_default_basis).get_info()
        basis.update({0: init_basis(stem_info['init_vec'], main_axis='y')})
        neurite.update({i: stem_info})
        cand_segs = set(stem['children'])
        if len(cand_segs) > 0:
            branch_hist = {i: {'term': new_skl.vertices.shape[0] - 1, 'children': set(stem['children'])}}
            p_seg = stem['skl']
            c_seg = [get_seg(skl=skl, seg_dict=seg_dict, index=s, parent_seg_index=i)['skl'] for s in stem['children']]
            bif_info = Bifurcation_info(p_seg, c_segs=c_seg, order=1, context={'p': i, 'c': stem['children']}, type_id=axon_type_id, extract_exact=extract_exact)
            bifurcation.append(bif_info.get_info())
            basis.update({j: bs for j, bs in zip(cand_segs, bif_info.distribute_basis())})

        order = 2
        while len(untrsv_seg) > 0:
            for cand_idx in cand_segs.copy():
                for j, hist in branch_hist.items():
                    if cand_idx in hist['children']:
                        branch_node = hist['term']
                        p_index = j

                cand = get_seg(skl=skl, seg_dict=seg_dict, index=cand_idx, parent_seg_index=p_index)
                cand_skl = cand['skl']
                new_skl = add_seg(new_skl, seg=cand_skl, branch_pos_idx=branch_node)

                children = list(cand['children'])
                if len(children) > 0:
                    p_seg = cand_skl
                    c_seg = [get_seg(skl=skl, seg_dict=seg_dict, index=s, parent_seg_index=p_index)['skl'] for s in children]
                    bif_info = Bifurcation_info(p_seg, c_segs=c_seg, order=order, context={'p': cand_idx, 'c': children}, type_id=axon_type_id)
                    bifurcation.append(bif_info.get_info())
                    basis.update({j: bs for j, bs in zip(children, bif_info.distribute_basis())})

                new_skl_info = Neurite_seg_info(cand_skl, order=order, parent_basis=basis[cand_idx], extract_exact=extract_exact)
                neurite.update({cand_idx: new_skl_info.get_info()})

                for cnd in cand['children']:
                    cand_segs.add(cnd)
                branch_hist.update({cand_idx: {'term': new_skl.vertices.shape[0] - 1, 'children': set(cand['children'])}})
                cand_segs.remove(cand_idx)
                untrsv_seg.remove(cand_idx)
                segs_trsv.add(cand_idx)

            order += 1

    neurite_info = neurite
    g = Graph(bifurcation)
    axon_info = Axon_info(type_id=axon_type_id, graph=g, init_vec=stem_info['init_vec'], init_r=init_r).info
    skl_info = {'info': axon_info, 'segments': neurite_info, 'bifurcations': bifurcation}

    return skl_info


def extract_soma(soma: Skeleton) -> dict:
    return {'vertices': soma.vertices.tolist(), 'radii': soma.radii.tolist(), 'edges': soma.edges.tolist()}


def extract(skl: Skeleton, soma_type: int=1, interp: bool=False, extract_exact: bool=False) -> dict:
    if interp:
        skl = interp_skl(skl, gap=2.5)
    neurite_skls, soma_skl = detach_soma(skl, soma_type=soma_type)
    neurite_info = {}
    for i, axon in neurite_skls.items():
        axon = extract_axon(axon, soma_type=soma_type, extract_exact=extract_exact)
        neurite_info.update({i: axon})
        pass
    soma_info = extract_soma(soma_skl)
    return {'axons': neurite_info, 'soma': soma_info}


def interp_skl(skl: Skeleton, gap: float=2.5) -> Skeleton:
    """Interp skl."""
    new_skl = init_skl()
    if np.sum(skl.vertex_types == 1) > 0:
        axons, soma = detach_soma(skl=skl)
        new_skl.vertices = np.vstack((new_skl.vertices, soma.vertices[0]))
        new_skl.vertex_types = np.array([1])
        new_skl.radius = np.array([soma.radius[0]])
    else:
        axons = {i: s for i, s in enumerate(skl.components())}

    def interp_nodes(v1: np.array, v2: np.array) -> np.ndarray:
        """Interp nodes."""
        new_pts = np.zeros((0, 3))
        vec = v2 - v1
        num_pts = int(np.linalg.norm(vec) // gap) + 1
        for i in range(num_pts):
            new_pts = np.vstack((new_pts, v1 + i * gap / np.linalg.norm(vec) * vec))
        return new_pts

    def interp_radius(v1: np.array, v2: np.array, r1: float, r2: float) -> List:
        """Interp radius."""
        new_r = []
        vec = v2 - v1
        r_diff = r2 - r1
        num_pts = int(np.linalg.norm(vec) // gap) + 1
        for i in range(num_pts):
            new_r.append(r1 + i * gap / np.linalg.norm(vec) * r_diff)
        return new_r

    def get_parent_path(paths: list, p: int) -> int:
        """Return the parent path."""
        start_node = paths[p][0]
        parent = []
        for i, path in enumerate(paths):
            if path[-1] == start_node:
                parent.append(i)
        if len(parent) == 0:
            return -1
        else:
            return parent[0]

    for axon_id, axon in axons.items():
        new_axon = init_skl()
        if np.sum(skl.vertex_types == 1) > 0:
            ori_vertices = axon.vertices.copy()[1:]
            ori_radius = axon.radius.copy()[1:]
            axon.edges = axon.edges[1:] - 1
        else:
            ori_vertices = axon.vertices.copy()
            ori_radius = axon.radius.copy()
            axon.edges = axon.edges.copy()

        paths = axon.interjoint_paths(return_indices=True)
        for i, path in enumerate(paths):
            seg_vertices = np.vstack([ori_vertices[k] for k in path])
            seg_radius = np.array([ori_radius[k] for k in path])
            interp_seg_pts = np.zeros((0, 3))
            interp_seg_r = np.array([])

            for j in range(0, seg_vertices.shape[0] - 1):
                interp_pts = interp_nodes(seg_vertices[j], seg_vertices[j + 1])
                if j + 2 < seg_vertices.shape[0] and len(interp_pts) > 1:
                    if np.linalg.norm(interp_pts[-1] - seg_vertices[j + 2]) < 0.5 * gap:
                        interp_pts = interp_pts[:-1]
                interp_seg_pts = np.vstack((interp_seg_pts, interp_pts))
                interp_r = interp_radius(seg_vertices[j], seg_vertices[j + 1], seg_radius[j], seg_radius[j + 1])
                interp_seg_r = np.concatenate((interp_seg_r, interp_r))

            interp_seg_pts = np.vstack((interp_seg_pts, seg_vertices[-1]))
            interp_seg_r = np.append(interp_seg_r, interp_seg_r[-1])
            num__ = new_axon.vertices.shape[0]

            parent = get_parent_path(paths=paths, p=i)
            if parent >= 0:
                interp_seg_pts = interp_seg_pts[1:]
                interp_seg_r = interp_seg_r[1:]
                interp_seg = single_neurite(vertices=interp_seg_pts)
                tail_parent_idx = paths[parent][-1]
                tail_parent_pt = ori_vertices[tail_parent_idx]
                tail_new_idx = np.argmin(cdist(new_axon.vertices, np.array([tail_parent_pt])))
                new_axon.edges = np.vstack((new_axon.edges, [tail_new_idx, num__], interp_seg.edges + num__))
            else:
                interp_seg = single_neurite(vertices=interp_seg_pts)
                new_axon.edges = np.vstack((new_axon.edges, interp_seg.edges + num__))

            new_axon.vertices = np.vstack((new_axon.vertices, interp_seg.vertices))
            new_axon.radius = np.concatenate((new_axon.radius, interp_seg_r))
            new_axon.vertex_types = np.concatenate((new_axon.vertex_types, np.ones(interp_seg.vertices.shape[0]) * axon.vertex_types[-1]))

        num_ = new_skl.vertices.shape[0]
        new_skl.vertices = np.vstack((new_skl.vertices, new_axon.vertices))
        new_skl.radius = np.concatenate((new_skl.radius, new_axon.radius))
        new_skl.vertex_types = np.concatenate((new_skl.vertex_types, new_axon.vertex_types)).astype(int)
        new_skl.edges = np.vstack((new_skl.edges, [0, num_], num_ + new_axon.edges))

    return new_skl


def get_ctrl_pts(raw_nrt: Skeleton, reg_basis: dict=None) -> Tuple:
    """Return the ctrl pts."""
    if raw_nrt.vertices.shape[0] < 5:
        total_len = np.sum([norm(raw_nrt.vertices[i] - raw_nrt.vertices[i + 1]) for i in range(len(raw_nrt.vertices) - 1)])
        gap = total_len / 6
        raw_nrt = interp_skl(raw_nrt, gap=gap)

    raw_pts = raw_nrt.vertices - raw_nrt.vertices[0]
    x, y, z = raw_pts.T
    tck, u = scipy.interpolate.splprep([x, y, z], s=0.0)
    t, ctrl_pts, k = (tck[0], np.array(tck[1]).T, tck[2])

    phi_list = []
    theta_list = []
    norm_list = []

    basis = _default_basis
    for i in range(1, ctrl_pts.shape[0]):
        vec = ctrl_pts[i] - ctrl_pts[i - 1]
        norm_list.append(norm(vec))
        vec = vec / norm(vec)
        phi, theta = get_sph_angles(carte=vec, basis=basis, order='yzy')
        phi_list.append(phi)
        theta_list.append(theta)
        basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=phi, gamma=theta)

    return (t, (phi_list, theta_list), k), norm_list, reg_basis


if __name__ == '__main__':
    pass
