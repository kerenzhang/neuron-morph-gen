from typing import List
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from cloudvolume import Skeleton
from components.neurite import Neurite
from typing import Union
import bezier
from scipy.interpolate import splprep, splev
import scipy
from utils.params_var import get_dist
from utils.utils import *

_default_basis = {'x': np.array([1.0, 0.0, 0.0]), 'y': np.array([0.0, 1.0, 0.0]), 'z': np.array([0.0, 0.0, 1.0])}
__all__ = ['radius_var', 'local_detail', 'soma_stem']


def asym_thickening(vertices: np.ndarray=None, magn: float=0.0, res: int=1) -> None:
    vecs = np.array([vertices[i + 1] - vertices[i] for i in range(len(vertices) - 1)])
    init_vec = vecs[-1]
    ecld_dist = np.sum(vecs[:-1])
    total_angle = magn * 2
    arc_len = 2 * (np.pi / 2 - magn) / (2 * np.pi) * (2 * np.pi * (ecld_dist / 2) * np.tan(magn))
    indv_angle = np.random.rand(res - 1) / (res - 1)
    indv_angle = indv_angle / np.sum(indv_angle) * total_angle
    indv_len = np.random.rand(res) / res
    indv_len = indv_len / np.sum(indv_len) * arc_len
    xyz = vecs.T
    tck, u = splprep(x=xyz, s=0)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new, z_new = splev(u_new, tck)
    # TODO: find proper asymmetric thickening function

def brownian_radius_variation(radius_list: List, **kwargs) -> np.ndarray:
    """
    Stochastic brownian radius variation
    """
    prob_params = {'freq': 0.3, 'mag': 0.6, 'len_low': 4, 'len_high': 12}
    prob_params.update(kwargs)
    radius_list = np.array(radius_list)
    total_len = len(radius_list)
    low, high = (int(prob_params['len_low']), int(prob_params['len_high']))

    covered = 0
    begin = []
    while covered < total_len * prob_params['freq']:
        segment_len = np.random.randint(low=low, high=high)
        if segment_len >= total_len:
            bgn_pt = 0
            segment_len = total_len
        else:
            bgn_pt = np.random.randint(low=0, high=total_len - segment_len)
        variation = brownian_path(1, N=segment_len)[0] * prob_params['mag']
        ori_r = radius_list[bgn_pt:bgn_pt + segment_len]
        new_r = ori_r + variation
        radius_list[bgn_pt:bgn_pt + segment_len] = new_r
        begin.append(bgn_pt)
        covered += segment_len

    radius_list = np.abs(radius_list)
    return radius_list

def random_thickening(radius_list: np.ndarray, vertices: np.ndarray, **kwargs) -> np.ndarray:
    """
    Random thickening of single neurite based on bezier-curves.
    """
    prob_params = {'freq': 0.3, 'mag': 0.8, 'len_low': 4, 'len_high': 12}
    prob_params.update(kwargs)
    radius_list = np.array(radius_list)
    seg_vec = np.diff(vertices, axis=0)
    total_len = len(radius_list)
    low, high = (int(prob_params['len_low']), int(prob_params['len_high']))

    covered = 0
    begin = []
    while covered < len(radius_list) * prob_params['freq']:
        segment_len = np.random.randint(low=low, high=high)
        if segment_len >= total_len:
            bgn_pt = 0
            segment_len = total_len
        else:
            bgn_pt = np.random.randint(low=0, high=total_len - segment_len)
        if len(begin) > 1:
            if np.abs(bgn_pt - begin[-1]) < segment_len:
                continue
        ori_r = radius_list[bgn_pt:bgn_pt + segment_len]
        seg = seg_vec[bgn_pt:bgn_pt + segment_len - 1]
        eucl_vec = [np.linalg.norm(v) for v in seg]

        # TODO find proper control points floating function
        seg = np.cumsum(eucl_vec) / np.sum(eucl_vec)
        mid_pt = np.random.rand() * 0.25 + 0.35
        left_1 = 1 / 3 * mid_pt
        left_2 = 2 / 3 * mid_pt
        right_1 = mid_pt + 1 / 3 * (1 - mid_pt)
        right_2 = mid_pt + 2 / 3 * (1 - mid_pt)

        bezier_x = np.array([0, left_1, left_2, mid_pt, right_1, right_2, 1])
        bezier_y = np.array([0, 0, 1, 1, 1, 0, 0]) * np.mean(radius_list) * prob_params['mag']
        nodes = np.asfortranarray([bezier_x, bezier_y])
        bezier_curve = bezier.curve.Curve(nodes, degree=6)

        pts_mid = seg[:-1]
        pts_mid = np.array(pts_mid).astype(np.double)

        add_radius = bezier_curve.evaluate_multi(pts_mid)[1]
        new_r = ori_r[1:-1] + np.array(add_radius)
        radius_list[bgn_pt + 1:bgn_pt + segment_len - 1] = new_r

        begin.append(bgn_pt)
        covered += segment_len

    return radius_list

def random_thinning(radius_list: np.ndarray, vertices: np.ndarray, **kwargs) -> np.ndarray:
    """
    Random thinning based on bezier-curves.
    """
    prob_params = {'freq': 0.5, 'mag': 0.5, 'len_low': 4, 'len_high': 12}
    prob_params.update(kwargs)
    radius_list = np.array(radius_list)
    seg_vec = np.diff(vertices, axis=0)
    total_len = len(radius_list)
    low, high = (int(prob_params['len_low']), int(prob_params['len_high']))

    covered = 0
    begin = []
    while covered < len(radius_list) * prob_params['freq']:
        segment_len = np.random.randint(low=low, high=high)
        if segment_len >= total_len:
            bgn_pt = 0
            segment_len = total_len
        else:
            bgn_pt = np.random.randint(low=0, high=total_len - segment_len)
        if len(begin) > 1:
            if np.abs(bgn_pt - begin[-1]) < segment_len:
                continue
        ori_r = radius_list[bgn_pt:bgn_pt + segment_len]
        seg = seg_vec[bgn_pt:bgn_pt + segment_len - 1]
        eucl_vec = [np.linalg.norm(v) for v in seg]
        seg = np.cumsum(eucl_vec) / np.sum(eucl_vec)

        mid_pt = np.random.rand() * 0.25 + 0.35
        left_1 = 1 / 3 * mid_pt
        left_2 = 2 / 3 * mid_pt
        right_1 = mid_pt + 1 / 3 * (1 - mid_pt)
        right_2 = mid_pt + 2 / 3 * (1 - mid_pt)

        bezier_x = np.array([0, left_1, left_2, mid_pt, right_1, right_2, 1])
        bezier_y = np.array([0, 0, 1, 1, 1, 0, 0]) * np.mean(radius_list) * prob_params['mag']
        nodes = np.asfortranarray([bezier_x, bezier_y])
        bezier_curve = bezier.curve.Curve(nodes, degree=6)

        pts_mid = seg[:-1]
        pts_mid = np.array(pts_mid).astype(np.double)

        add_radius = bezier_curve.evaluate_multi(pts_mid)[1]
        new_r = ori_r[1:-1] - np.array(add_radius)
        radius_list[bgn_pt + 1:bgn_pt + segment_len - 1] = new_r

        begin.append(bgn_pt)
        covered += segment_len

    return np.abs(radius_list)

def radius_var(skl: Skeleton, **kwargs) -> Skeleton:
    seq = [random_thickening, random_thinning, brownian_radius_variation]
    for s_o in seq:
        vv = skl.vertices
        rr = skl.radius
        skl.radius = s_o(radius_list=rr, vertices=vv, **kwargs)
    if 'smooth' in kwargs.keys():
        smooth = kwargs['smooth']
    else:
        smooth = 1.5
    if 'smooth_r' in kwargs.keys():
        smooth_r = int(kwargs['smooth_r'])
    else:
        smooth_r = 2
    skl.radius = scipy.ndimage.gaussian_filter1d(skl.radius, smooth, radius=smooth_r)

    return skl

def local_detail(neurite: Neurite, **kwargs) -> Skeleton:

    prob_params = {'freq': 0.2, 'radius': 1.0, 'len_scale': 1}
    prob_params.update(kwargs)
    skl = neurite.skl.clone()
    new_neurite = copy.deepcopy(neurite)
    seq = [dendritic_spines]

    for s_o in seq:
        num = skl.vertices.shape[0]
        cand_num = np.floor((num - 2) * prob_params['freq']).astype(int)
        if cand_num > 1:
            while True:
                cand = list(range(2, num - 2))
                random.shuffle(cand)
                cand_idx = np.sort(cand[:cand_num])
                if np.min(np.diff(cand_idx)) > 0.5 / prob_params['freq']:
                    break
            for c_i in cand_idx:
                vert_insert = skl.vertices[c_i - 1:c_i + 2]
                basis = neurite.basis_hist[c_i]
                detail_skl = s_o(vertices=vert_insert, basis=basis, radius=prob_params['radius'], len_scale=prob_params['len_scale'])
                num_nodes = skl.vertices.shape[0]
                skl.vertices = np.vstack((skl.vertices, detail_skl.vertices))
                skl.vertex_types = np.concatenate((skl.vertex_types, detail_skl.vertex_types))
                skl.radius = np.concatenate((skl.radius, detail_skl.radius))
                new_edges = detail_skl.edges + num_nodes
                skl.edges = np.vstack((skl.edges, [c_i, num_nodes], new_edges))

    new_neurite.skl = skl
    return new_neurite


def dendritic_spines(vertices: np.ndarray, basis: dict, radius: float=1.0, len_scale: float=1.0) -> Skeleton:

    phi = get_dist('uniform', 1, low=np.pi / 4, high=np.pi * 3 / 4)
    theta = get_dist('uniform', 1, low=-np.pi, high=np.pi)
    v0 = vertices[1]

    stem_vec = vec_from_sph(phi=phi, theta=theta, basis=basis, order='yzy')
    stem_vec = stem_vec * get_dist('uniform', 1, low=2, high=5)
    spline_nodes = v0 + stem_vec

    basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=phi, gamma=theta)
    sign = 1
    if np.random.rand() < 0.5:
        sign = -1

    first_angle = get_dist('uniform', 1, low=-np.pi / 4, high=np.pi / 4)
    first_vec = vec_from_sph(phi=first_angle, theta=0, order='yzy', basis=basis)
    first_vec *= get_dist('uniform', 1, low=1.2, high=1.6)
    spline_nodes = np.vstack((spline_nodes, spline_nodes + first_vec))

    basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=first_angle, gamma=0)
    num = 4
    next_angles = get_dist('uniform', num, np.pi / 2, np.pi / 6) * sign
    for i in range(num):
        vec = vec_from_sph(phi=next_angles[i], theta=0, basis=basis, order='yzy')
        vec *= get_dist('uniform', 1, low=0.8, high=1.2)
        spline_nodes = np.vstack((spline_nodes, spline_nodes[-1] + vec))
        basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=next_angles[i], gamma=0)

    spline_vertices = interp_spline(spline_nodes, sample_num=num * 2)
    v_types = np.ones(spline_vertices.shape[0]) * 9
    r = np.ones(spline_vertices.shape[0]) * radius * len_scale
    edges = np.vstack([[i, i + 1] for i in range(spline_vertices.shape[0] - 2)])
    spline_skl = Skeleton(vertices=spline_vertices, edges=edges, radii=r, vertex_types=v_types)

    return spline_skl

def soma_stem(stem: Neurite, max_r: float=10.0, period: float=10.0) -> Neurite:

    stem_skl = stem.skl.clone()
    new_stem = copy.deepcopy(stem)
    len_vec = [np.linalg.norm(stem_skl.vertices[i + 1] - stem_skl.vertices[i]) for i in range(len(stem_skl.vertices) - 1)]
    if period > np.sum(len_vec):
        period = np.sum(len_vec) / 2
    threshold = 0.1
    decay_coef = -np.log(threshold) / period
    res = 1.0
    cut_off = np.where(np.cumsum(len_vec) > period)[0][0] + 1
    init_r = stem_skl.radius[cut_off]
    if max_r < init_r:
        max_r = init_r * 2
    new_pts = np.array([stem_skl.vertices[0]])
    new_r = np.array([max_r])
    t = 0

    def sample_r(time) -> float:
        """Sample r."""
        return (max_r - init_r) * np.exp(-decay_coef * time) + init_r

    for i in range(0, cut_off):
        num_pts = int(len_vec[i] // res)
        vec = stem_skl.vertices[i + 1] - stem_skl.vertices[i]
        vec = vec / np.linalg.norm(vec)
        for j in range(1, num_pts):
            t += j * res
            new_pos = stem_skl.vertices[i] + vec / np.linalg.norm(vec) * res * j
            new_pts = np.vstack([new_pts, new_pos])
            new_r = np.append(new_r, sample_r(time=t))
        t = np.cumsum(len_vec)[i]
        new_pts = np.vstack([new_pts, stem_skl.vertices[i + 1]])
        new_r = np.append(new_r, sample_r(time=t))

    truncated_vertices = stem_skl.vertices[cut_off + 1:]
    truncated_r = stem_skl.radius[cut_off + 1:]

    final_vertices = np.vstack((new_pts, truncated_vertices))
    final_r = np.concatenate((new_r, truncated_r))
    final_v_types = np.concatenate((np.ones(new_pts.shape[0]) * 7, np.ones(truncated_vertices.shape[0]) * stem_skl.vertex_types[0]))
    final_edges = np.vstack([[i, i + 1] for i in range(final_vertices.shape[0] - 1)])

    new_skl = Skeleton(vertices=final_vertices, edges=final_edges, radii=final_r, vertex_types=final_v_types)
    new_stem.skl = new_skl

    return new_stem



