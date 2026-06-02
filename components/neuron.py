import copy
from cloudvolume import Skeleton
import numpy as np
from numpy.linalg import norm
import pickle
from typing import Union, Tuple, List
import os
from components.base_classes import Node, Element
from components.neurite import Neurite, Axon
from components.soma import SOMA
from utils.utils import *
from utils.params_var import *
from utils.postprocessing import *
from utils.graph import random_tree_gen, interp_graph


class Neuron(Element):
    """
    A general neuron class that contains growth functions for subcomponents.
    """

    def __init__(self, template: str='', skl: Skeleton=None, **kwargs) -> None:
        """
        Create a neuron from a template        :
        param template: a template file path to begin with
        """
        super(Neuron, self).__init__()
        self.space = 'physical'
        self.soma_template = {}
        self.neurite_template = {}
        self.template = template
        if not template:
            self.session_name = 'new test'
            if 'name' in kwargs:
                self.session_name = kwargs['name']
        if not os.path.isdir(template):
            raise FileNotFoundError('The template was not found at: {}.'.format(template))

        self.elements = {'soma': [], 'neurite': {}}
        self.all_graphs = {}
        self.all_skl = {}
        if os.path.isdir(os.path.join(self.template, 'soma')) and os.listdir(os.path.join(self.template, 'soma')):
            with open(os.path.join(self.template, 'soma', 'soma.pkl'), 'rb') as f:
                soma_info = pickle.load(f)
            self.soma_type = soma_info['type']
            self.soma_volume = soma_info['volume']
            self.soma_noise = soma_info['noise']
            self.ctr = np.array(soma_info['ctr'])
        else:
            # default random soma
            self.soma_type = 'sphere'
            self.soma_volume = (10000, 1000)
            self.soma_noise = (0.2, 0.05)
            self.ctr = np.array([0, 0, 0])
        self.soma_stem = {}
        if os.path.isdir(os.path.join(self.template, 'post-processing')) and os.listdir(os.path.join(self.template, 'post-processing')):
            with open(os.path.join(self.template, 'post-processing', 'pp.pkl'), 'rb') as f:
                self.pp = pickle.load(f)
        if skl is not None:
            self.skl = skl

    def create(self, updated_params: neuron_params=None, create_exact: bool=True) -> None:
        """
        hierarchy diagram: axon_type -> individual axon -> neurite
        components, branches and bifurcations.
        Create an instance of neuron with one sampling process of each parameter.
        If the create_mode is exact, it will follow the template exactly in terms of line curvature and overall graph.
        If not, it will draw from distribution.
        """
        if not updated_params:
            nnpm = neuron_params(os.path.join(self.template, 'params'))
        else:
            nnpm = updated_params
        nnpm.initialize_axons()
        axons = nnpm.axons

        for type_id, axon_types in axons.items():
            for axon_id, axon in axon_types.items():
                neurite_segs = {}
                bif_list = []
                num_seg = 0
                # Load graph parameters if any. If not, generate random graph.
                if os.listdir(os.path.join(self.template, 'graphs')):
                    with open(interp_graph(path=os.path.join(self.template, 'graphs'), type_id=type_id, max_order=nnpm.get_params('max_order', type_id, 1).sample(1), num_segments=nnpm.get_params('num_segments', type_id, 1).sample(1)), 'rb') as f:
                        _graph = pickle.load(f)
                else:
                    _graph = random_tree_gen(scheme=nnpm.get_choices('order_scheme', type_id, 1).values, max_order=nnpm.get_params('max_order', type_id, 1).sample(1), num_segments=nnpm.get_params('num_segments', type_id, 1).sample(1), symmetry=nnpm.get_params('symmetry', type_id, 1).sample(1), d2h_ratio=nnpm.get_params('d2h_ratio', type_id, 1).sample(1))
                _max_order, _num_segments = (_graph.max_order, _graph.num_segments)

                seg_list = dict([(j, []) for j in range(1, _max_order + 1)])
                # BFS traversal for each neurite at different level orders.
                for i in range(1, _max_order + 1):
                    for seg_id in _graph.trs_steps()[i]:
                        from_soma = True if seg_id == 0 else False
                        start = np.array([0, 0, 0])
                        branch_attrs = {'start': Node(start), 'neurite_id': seg_id, 'order': i, 'from_soma': from_soma, 'from_spline': create_exact}

                        if type_id == 4: # basal dendrite is unique (might be shaft order scheme)
                            order = _graph.get_order(idx=seg_id, scheme='s')
                        else:
                            order = i

                        # branch segment level
                        segment_lvl = ['init_r', 'init_vec', 'proj_w', 'hist_w', 'contr_factor']
                        segment_lvl_param = {p: nnpm.get_params(p, type_id=type_id, order=order).sample(1) for p in segment_lvl}
                        if not create_exact:
                            _length = {'length': nnpm.get_params('length', type_id=type_id, order=order).sample(1)}
                        else:
                            _length = {'length': nnpm.get_params('length', type_id=type_id, order=order, target_id=seg_id).sample(1)}
                        segment_lvl_param.update(_length)

                        # node level
                        step_size_mean, step_size_var = nnpm.get_params('step_size', type_id=type_id, order=order).values
                        steps = get_steps(length=segment_lvl_param['length'], step_mean=step_size_mean, step_var=step_size_var)
                        num_conpts = len(steps)

                        if 'spl_ctrl_pts' in nnpm.params and create_exact:
                            tck, len_vec, reg_basis = nnpm.get_params(param_name='spl_ctrl_pts', type_id=type_id, order=order, target_id=seg_id).values
                            t, ctrl_pts, k = tck
                            phi_noise = np.zeros(len(ctrl_pts[0]))
                            theta_noise = np.zeros(len(ctrl_pts[0]))
                            ctrl_pts_phi = np.array(ctrl_pts[0]) + np.array(phi_noise)
                            ctrl_pts_theta = np.array(ctrl_pts[1]) + np.array(theta_noise)
                            phi_list, theta_list = Neurite.from_spline(t=t, ctrl_pts=(ctrl_pts_phi, ctrl_pts_theta), k=k, norm_list=len_vec, step_sizes=steps, reg_basis=reg_basis)
                        else:
                            phi_list = nnpm.get_params(param_name='phi', type_id=type_id, order=order).sample(num_conpts)
                            theta_list = nnpm.get_params(param_name='theta', type_id=type_id, order=order).sample(num_conpts)
                            if num_conpts == 1:
                                phi_list, theta_list = ([phi_list], [theta_list])
                            reg_basis = None

                        angles = [(p, t) for p, t in zip(phi_list, theta_list)]

                        node_level_param = {'step_length': steps, 'rel_angles': angles, 'reg_basis': reg_basis}
                        n = {**branch_attrs, **node_level_param, **segment_lvl_param}
                        seg_list[i].append(num_seg)
                        neurite_segs.update({seg_id: n})

                        # pass the current bifurcation geometry to children
                        c_cand = _graph.get_children(idx=seg_id)
                        if c_cand.size > 0:
                            bif_attrs = {'order': i, 'context': {'p': seg_id, 'c': c_cand}}
                            if create_exact:
                                reverse = nnpm.get_params(param_name='reverse', type_id=type_id, order=order, target_id=seg_id).values
                                bif_attrs.update({'reverse': reverse})
                            bif_lvl = ['ampl', 'tilt', 'roll', 'ratio']
                            bif_params = {p: nnpm.get_params(param_name=p, type_id=type_id, order=order).sample(1) for p in bif_lvl}
                            bif_dict = {**bif_attrs, **bif_params}
                            bif_list.append(bif_dict)

                axon_comp = {'segments': neurite_segs, 'bifs': bif_list}
                if int(type_id) not in self.neurite_template.keys():
                    self.neurite_template.update({int(type_id): {}})
                type_axon = self.neurite_template[int(type_id)]
                type_axon.update({int(axon_id): axon_comp})
                self.neurite_template[int(type_id)] = type_axon

    def from_scratch(self, name: str='0') -> None:
        """
        Generate a neuron from scratch without templates.
        """
        os.makedirs('swc', exist_ok=True)
        os.makedirs(os.path.join('swc', name), exist_ok=True)
        for f in ['params', 'soma', 'graphs', 'post-processing', 'generated']:
            os.makedirs(os.path.join('swc', name, f), exist_ok=True)
        self.template = os.path.join('swc', name)

    def attach_soma(self, soma: SOMA, conn_node: Node=None) -> None:
        """
        Attach soma to the generated axons
        """
        if not conn_node:
            self.vertices = soma.vertices
            self.edges = soma.edges
            self.vertex_types = soma.vertex_types
            self.nodes.extend(soma.nodes)
            self.radius = soma.radius
        else:
            pre_num = self.num_nodes
            new_vertices = soma.vertices
            for n_node in soma.nodes:
                n_node.id += pre_num
            new_edges = soma.edges + pre_num
            self.vertices = np.vstack((self.vertices, new_vertices))
            self.vertex_types = np.concatenate((self.vertex_types, soma.vertex_types))
            self.radius = np.concatenate((self.radius, soma.radius))
            self.edges = np.vstack((self.edges, np.array([conn_node.id, pre_num]), new_edges))
            self.nodes.extend(soma.nodes)
        self.elements['soma'].append(soma)

    def attach_neurite(self, parent_axon_id: int, new_neurite: Neurite, conn: dict) -> None:
        """
        Attach neurite to the generated axons
        """
        parent_axon = self.all_skl[parent_axon_id]
        num_nodes_b4 = parent_axon.skl.vertices.shape[0]
        last_main_node_idx = np.where(new_neurite.skl.vertex_types < 9)[0][-1]

        new_vertices = np.vstack((parent_axon.skl.vertices, new_neurite.skl.vertices))
        new_vertex_types = np.concatenate((parent_axon.skl.vertex_types, new_neurite.skl.vertex_types))
        new_radius = np.concatenate((parent_axon.skl.radius, new_neurite.skl.radius))

        edges_added = new_neurite.skl.edges.copy()
        edges_added += num_nodes_b4
        new_edges = np.vstack((parent_axon.skl.edges, edges_added))

        for p in conn['p']:
            p_idx = self.find_node_idx(parent_axon.skl.vertices, p)
            new_edges = np.vstack((new_edges, [p_idx, num_nodes_b4]))
        for c in conn['c']:
            c_idx = self.find_node_idx(parent_axon.skl.vertices, c)
            new_edges = np.vstack((new_edges, [last_main_node_idx + num_nodes_b4, c_idx]))

        new_skl = Skeleton(vertices=new_vertices, edges=new_edges, vertex_types=new_vertex_types, radii=new_radius)
        parent_axon.skl = new_skl
        self.all_skl[parent_axon_id] = parent_axon

    def delete_neurite(self, neurite_id: Tuple[int, int]) -> dict:
        """
        Delete neurite.
        """
        axon_id, seg_id = neurite_id
        neurite_del = self.elements['neurite'][axon_id][seg_id]
        parent_axon = self.all_skl[axon_id]
        graph = self.all_graphs[axon_id]
        prt_neurite = graph.get_parent(seg_id)
        child_neurite = graph.get_children(seg_id)

        left_nodes = {'c': [], 'p': []}
        if len(child_neurite) > 0:
            end_node = neurite_del.skl.vertices[-1]
            end_idx = self.find_node_idx(parent_axon.skl.vertices, end_node)
            c_idx = self.find_node_neighbors(parent_axon.skl, end_idx)
            left_nodes.update({'c': [c for c in c_idx if not self.is_in_seg(neurite_del.skl, c)]})
        if prt_neurite > -1:
            start_node = neurite_del.skl.vertices[0]
            start_idx = self.find_node_idx(parent_axon.skl.vertices, start_node)
            p_idx = self.find_node_neighbors(parent_axon.skl, start_idx)
            left_nodes.update({'p': [p for p in p_idx if not self.is_in_seg(neurite_del.skl, p)]})

        new_axon = copy.deepcopy(parent_axon)
        for v in neurite_del.skl.vertices:
            new_axon.skl = self.delete_node(new_axon.skl, node=v)
        self.all_skl[axon_id] = new_axon

        return left_nodes

    def replace_neurite(self, old_neurite: Tuple[int, int], new_neurite: Neurite) -> None:
        """
        Replace neurite with a new neurite.
        """
        left_conn = self.delete_neurite(old_neurite)
        axon_id = old_neurite[0]
        new_neurite_copy = copy.deepcopy(new_neurite)
        self.attach_neurite(axon_id, new_neurite_copy, conn=left_conn)
        self.elements['neurite'][axon_id][old_neurite[1]] = new_neurite_copy

    def grow(self) -> dict:
        """
        Grow the neurite with pre-defined sequence.
        """
        for type_id, axon_type in self.neurite_template.items():
            for axon_id, axon_comp in axon_type.items():
                axon = Axon(comp=axon_comp, id=axon_id, type_id=type_id)
                axon.grow()
                self.elements['neurite'].update({axon_id: axon.grown_neurite})
                self.all_graphs.update({axon_id: axon.graph})
                self.all_skl.update({axon_id: axon})
        self.construct_soma()
        return self.all_skl

    def construct_soma(self) -> None:
        """Construct soma with given volume, shape, and noise.
        """
        soma_volume = get_dist('norm', 1, *self.soma_volume)
        soma_noise = get_dist('norm', 1, *self.soma_noise)
        soma = SOMA(shape=self.soma_type, volume=soma_volume, noise=soma_noise)
        self.elements['soma'].append(soma)

    def assemble(self) -> Skeleton:
        """
        Assemble the grown subcomponents and connect them into a complete neuron.
        """
        _soma_contact_node = 0
        r = 4.0
        soma_skl = Skeleton(vertices=np.array([self.ctr]), radii=np.array([r]), vertex_types=np.array([1]), edges=np.zeros((0, 2)))
        self.skl = soma_skl

        for i, skl_comp in self.all_skl.items():
            self.add_skl(new_skl=skl_comp.skl)

        stem_nodes = [sn.skl.vertices[0] for sn in self.find_neurite_property({'from_soma': True})]
        for j, sn in enumerate(stem_nodes):
            sn_idx = self.find_node_idx(self.skl.vertices, sn)
            self.skl.edges = np.vstack((self.skl.edges, [_soma_contact_node, sn_idx]))
            self.soma_stem.update({j: sn})

        return self.skl

    def affine_transform(self, scale: dict=None, angle_x: float=0.0, angle_y: float=0.0, angle_z: float=0.0) -> None:
        """
        Affine transform as part of post-processing
        """
        if scale is None:
            scale = {'x': 1.0, 'y': 1.0, 'z': 1.0, 'r': 1.0}

        def apply_to_axon(trans) -> None:
            """Apply to axon."""
            for axon_id, axon in self.all_skl.items():
                axon.skl.transform = trans
                axon.skl.apply_transform()
            ele = self.elements['neurite']
            for axon_id, axon_comp in ele.items():
                for neurite_id, nrt in axon_comp.items():
                    nrt.skl.transform = trans
                    nrt.skl.apply_transform()

        # Sequential rotation in yzx order for each axis (default basis)
        rot_xz = np.array([[np.cos(angle_y), 0, np.sin(angle_y), 0], [0, 1, 0, 0], [-np.sin(angle_y), 0, np.cos(angle_y), 0]])
        apply_to_axon(rot_xz)
        rot_xy = np.array([[np.cos(angle_z), -np.sin(angle_z), 0, 0], [np.sin(angle_z), np.cos(angle_z), 0, 0], [0, 0, 1, 0]])
        apply_to_axon(rot_xy)
        rot_zy = np.array([[1, 0, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x), 0], [0, np.sin(angle_x), np.cos(angle_x), 0]])
        apply_to_axon(rot_zy)

        # convert physical space (in micron) to voxel space for mask generation
        if self.space == 'physical':
            comp = np.array([[1 / scale['x'], 0, 0, 0], [0, 1 / scale['y'], 0, 0], [0, 0, 1 / scale['z'], 0]])
        else:
            comp = np.array([[scale['x'], 0, 0, 0], [0, scale['y'], 0, 0], [0, 0, scale['z'], 0]])

        comp = np.column_stack((comp[:, :3], self.ctr))
        apply_to_axon(comp)

        soma_pts = single_neurite(vertices=self.elements['soma'][-1].vertices)
        soma_pts.transform = comp
        soma_pts.apply_transform()
        self.elements['soma'][-1].vertices = soma_pts.vertices

        radius_scale = norm(comp[:, :3]) / np.sqrt(3) * scale['r']
        self.skl.radius *= radius_scale

        for axon_id, axon in self.all_skl.items():
            axon.skl.radius *= radius_scale

    def find_terminal(self, confine_neurite: Neurite=None) -> List:
        """Find terminal node of each axon"""
        if not confine_neurite:
            return self.terminals()
        else:
            terminals = []
            neurite_id_list = [n.id for n in confine_neurite.nodes]
            for t in self.terminals():
                if t in neurite_id_list:
                    terminals.append(t)
            return terminals

    def find_neurite_property(self, property_value: dict) -> List:
        p, v = list(property_value.items())[0]
        assert hasattr(self.elements['neurite'][0][0], p), 'does not own this property'
        cand = []
        for skl_comp in self.elements['neurite'].values():
            for ind_seg in skl_comp.values():
                if getattr(ind_seg, p) == v:
                    cand.append(ind_seg)
        return cand

    def postprocessing(self, return_new: bool=True) -> Neurite:
        """
        Apply affine transformations, stochastic perturbations, and morphology addition (e.g. dendritic spines)
        as post-processing to grown neurons.
        """
        if return_new:
            target = copy.deepcopy(self)
        else:
            target = self

        pp = self.pp
        all_axons = self.elements['neurite']

        # stochastic radius variation
        if 'radius_var' in pp.keys():
            for axon_id, axon_comp in all_axons.items():
                for neurite_id, nrt in axon_comp.items():
                    prob_params = {}
                    for key, value in pp['radius_var'].items():
                        prob_sample = get_dist('uniform', 1, *value)
                        prob_params.update({key: prob_sample})
                    new_skl = radius_var(nrt.skl, **prob_params)
                    nrt.skl = new_skl
                    target.replace_neurite(old_neurite=(axon_id, neurite_id), new_neurite=nrt)

        # local morphological mutation. Adding strcuture such as dendritic spines
        if 'local_detail' in pp.keys():
            for axon_id, axon_comp in all_axons.items():
                for neurite_id, nrt in axon_comp.items():
                    if neurite_id >= 2:
                        prob_params = {}
                        for key, value in pp['local_detail'].items():
                            prob_sample = get_dist('uniform', 1, *value)
                            prob_params.update({key: prob_sample})
                        new_nrt = local_detail(nrt, **prob_params)
                        target.replace_neurite(old_neurite=(axon_id, neurite_id), new_neurite=new_nrt)

        # transition from soma to attached neurite
        if 'soma_stem' in pp.keys():
            for axon_id, axon_comp in all_axons.items():
                new_stem = soma_stem(axon_comp[0], max_r=pp['soma_stem']['max_r'], period=pp['soma_stem']['period'])
                target.replace_neurite(old_neurite=(axon_id, 0), new_neurite=new_stem)

        # affine transformation
        if 'rot' in pp.keys():
            angle_x, angle_y, angle_z = map(target.pp['rot'].__getitem__, ('x', 'y', 'z'))
            angle_x = get_dist('uniform', 1, *angle_x)
            angle_y = get_dist('uniform', 1, *angle_y)
            angle_z = get_dist('uniform', 1, *angle_z)
            rot_info = {'angle_x': angle_x, 'angle_y': angle_y, 'angle_z': angle_z}
        else:
            rot_info = {'angle_x': 0, 'angle_y': 0, 'angle_z': 0}
        if 'scale' in pp.keys():
            scale_info = target.pp['scale']
        else:
            scale_info = {'x': 1, 'y': 1, 'z': 1}
        affine = {'scale': scale_info, **rot_info}
        target.affine_transform(**affine)

        return target

    def clear(self) -> None:
        self.__init__(name=self.session_name)

    def write_swc(self, swc_dir: str, flip_xz: bool=False) -> None:
        """
        flip xz is converting to numpy array format
        """
        if flip_xz:
            self.skl.vertices = np.flip(self.skl.vertices, axis=1)

        base_dir = os.path.dirname(swc_dir)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError('This directory does not exist: {}'.format(base_dir))

        with open(swc_dir, 'w') as f:
            f.write(self.skl.to_swc())

    def write_pkl(self, pkl_dir: str, flip_xz: bool=False) -> None:
        if flip_xz:
            self.skl.vertices = np.flip(self.skl.vertices, axis=1)

        base_dir = os.path.dirname(pkl_dir)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError('This directory does not exist: {}'.format(base_dir))

        with open(pkl_dir, 'wb') as f:
            pickle.dump(self, f)

    def __str__(self) -> str:
        verbose = 'Neuron with {} axons'.format(len(self.elements['neurite']))
        return verbose


