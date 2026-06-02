import numpy as np
from numpy.linalg import norm
from pathlib import Path
from cloudvolume import Skeleton
import scipy.interpolate as intp
import pickle
from typing import Iterable, Union, List, Tuple
from scipy.spatial.transform import Rotation

_default_basis = {'x': np.array([1.0, 0.0, 0.0]),
                  'y': np.array([0.0, 1.0, 0.0]),
                  'z': np.array([0.0, 0.0, 1.0])}
_axis_order = 'yxy'
_default_vec_order = 'zyz'
_vec_order = 'yzy'
_branch_order = 'yzy'
_numeric_zero = 1e2
__all__ = ['init_skl', 'init_basis', 'proj', 'residual', 'check_clockwise', 'axis_rot', 'order_switch',
           'get_sph_angles', 'vec_from_sph', 'update_basis', 'get_conn', 'get_angle', 'read_swc', 'read_neuron',
           'rall_rule', 'rot_basis', 'get_euler_angles', '_default_basis', 'get_steps', 'brownian_path',
           'interp_spline', 'single_neurite', 'write_swc', 'align_basis', 'combine_skl_no_conn', 'cord2skl',
           'find_internal_aux_vec', 'get_unit_vec']


def init_skl() -> Skeleton:
    """
    Initialize an empty skeleton.
    """
    vertices = np.zeros((0, 3), dtype=float)
    radii = np.array([])
    vertex_types = np.array([])
    edges = np.zeros((0, 2), dtype=int)
    return Skeleton(vertices=vertices, edges=edges, vertex_types=vertex_types, radii=radii)


def init_basis(main_vec: np.ndarray, aux_vec: np.ndarray=None, main_axis: str='y', from_default: bool=False) -> dict:
    """
    Initialize the coordinate basis based on rotation order and two orthogonal vectors.
    """
    basis = {}
    main_vec = main_vec / norm(main_vec)
    basis[main_axis] = main_vec
    if not from_default:
        if aux_vec is None:
            aux_vec = np.array([1.0, 0, 0])
            if get_angle(aux_vec, main_vec) < 0.1:
                aux_vec = np.array([0.0, 1, 0])
        aux_vec = residual(main_vec, aux_vec)
        if norm(aux_vec) < 0.01:
            aux_vec = np.array([1.0, 0.0, 0.0])
            aux_vec = residual(main_vec, aux_vec)
        aux_vec = aux_vec / norm(aux_vec)
        if main_axis == 'y':
            basis['x'] = aux_vec
            basis['z'] = np.cross(aux_vec, main_vec)
        elif main_vec == 'z':
            basis['x'] = aux_vec
            basis['y'] = np.cross(main_vec, aux_vec)
        else:
            basis['y'] = aux_vec
            basis['z'] = np.cross(main_vec, aux_vec)
    else:
        p, t = get_sph_angles(carte=main_vec, basis=_default_basis, order='yzy')
        basis = rot_basis(_default_basis, alpha=0, beta=p, gamma=t, intrinsic=False, order='yzy')
    return basis


def proj(basis: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Acquire the projection of a vector on each axis of a basis.
    """
    basis = basis / norm(basis)
    vec = vec / norm(vec)
    proj_vec = basis * np.dot(basis, vec)
    return proj_vec


def residual(basis: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Acquire the residual of the vector projection.
    """
    vec = vec / norm(vec)
    proj_vec = proj(basis, vec)
    return vec - proj_vec


def get_unit_vec(vec: np.ndarray) -> np.ndarray:
    """
    Return the unit vec.
    """
    vec = vec / norm(vec)
    return vec


def get_steps(length: float=1.0, step_mean: Union[int, float]=1.0, step_var: Union[int, float]=1.0) -> np.ndarray:
    """
    get steps of each increment of neurite growth based on random step size perturbation.
    """
    default_num = int(length // step_mean)
    step_list = step_mean + step_var * np.random.randn(default_num)
    current_len = np.sum(step_list)
    steps = step_list
    if current_len > length:
        while current_len > length:
            if np.sum(steps) > length:
                break
            else:
                steps = steps[:-1]
    else:
        while np.sum(steps) < length:
            adn_step = step_mean + step_var * np.random.randn()
            steps = np.append(steps, adn_step)
            if np.sum(steps) >= length:
                break
    steps = np.clip(steps, 0.1, 999)
    return steps


def check_clockwise(rot_axis: Union[np.ndarray, list], source: Union[np.ndarray, list], target: Union[np.ndarray, list],
                    angle: float=0.0) -> bool:
    """
    Check if the rotation direction is clockwise or not.
    """
    rot_result = axis_rot(u=rot_axis, v=source, angle=angle)
    if norm(rot_result - target) < 0.001:
        return False

    return True


def axis_rot(u: Union[np.ndarray, list], v: Union[np.ndarray, list], angle: float) -> np.ndarray:
    """
    u: rotating around this axis
    v: vector to be rotated
    The rotation is counterclockwise by default.
    """
    u = np.array(u) / norm(u)
    v = np.array(v)
    u_v = np.cos(angle) * v + np.sin(angle) * np.cross(u, v) + (1 - np.cos(angle)) * np.dot(u, v) * u
    return u_v


def order_switch(order: str, default_order: str='zyz') -> dict:
    """
    Switch the rotation order.
    """
    basis = {default_order[0]: order[0], default_order[1]: order[1]}
    remain_axis = [a for a in 'xyz' if a not in order][0]
    basis['x'] = remain_axis
    return basis


def get_sph_angles(carte: Union[np.ndarray, list], basis: dict=None, order: str='zyz') -> tuple:
    """
    Get phi and theta based on arbitrary orthogonal basis and axis order in spherical
    coordinates format
    :param carte: cartesian coordinates in x, y, z
    :param basis: the orthogonal basis that contains x, y, z axis vector
    :param order: the axis order that substitutes the traditional zyz order
    :return: theta and phi
    """
    carte = carte / norm(carte)
    assert len(order) == 3 and order[0] == order[2], 'please enter the correct order'
    if not basis:
        basis = {c: t for c, t in zip('xyz', np.identity(3))}
    if order == 'yzy':
        z = np.dot(carte, basis['z'])
        y = np.dot(carte, basis['y'])
        x = np.dot(carte, basis['x'])
        x, y, z = (z, x, y)
    else:
        z = np.dot(carte, basis['z'])
        y = np.dot(carte, basis['y'])
        x = np.dot(carte, basis['x'])
    phi = np.arctan2(np.sqrt(x * x + y * y), z)
    theta = np.arctan2(y, x + 0.001)
    return phi, theta


def vec_from_sph(phi: float=0.0, theta: float=0.0, basis: dict=None, order: str='zyz') -> np.ndarray:
    """
    Acquire the vector from spherical coordinates.
    """
    if not basis:
        basis = {c: t for c, t in zip('xyz', np.identity(3))}
    assert len(order) == 3 and order[0] == order[2], 'please enter the correct order'
    if order == 'yzy':
        new_basis = {'x': basis['z'], 'y': basis['x'], 'z': basis['y']}
    else:
        new_basis = basis
    vec = new_basis['z']
    vec = axis_rot(new_basis['y'], vec, phi)
    vec = axis_rot(new_basis['z'], vec, theta)
    return vec


def update_basis(ori_basis: dict, phi: float=0.0, theta: float=0.0, order: str='zyz') -> dict:
    """
    Update the basis using euler angles and the corresponding rotation order.
    """
    x, y, z = (ori_basis['x'], ori_basis['y'], ori_basis['z'])
    new_basis = {'x': x, 'y': y, 'z': z}
    angles = [phi, theta]
    if order[-1] == 'y':
        order = 'yxy'
    for o, ang in zip(order[1:], angles):
        for ax in ['x', 'y', 'z']:
            new_basis[ax] = axis_rot(ori_basis[o], new_basis[ax], ang)
    return new_basis


def EMA(vec_list: np.ndarray) -> np.ndarray:
    """
    Exponential moving average of a vector.
    """
    weight = np.array([0.75, 0.25, 0.1, 0.05, 0.05])
    n_vec = len(vec_list)
    weight_range = weight[:n_vec]
    return np.squeeze(np.sum([[w * v] for w, v in zip(weight_range, vec_list)], axis=0) / norm(weight_range))


def get_conn(node_id: int, edge_list: np.ndarray) -> bool:
    """
    Check if a node is in the edge list.
    """
    return np.sum(edge_list == node_id)


def get_angle(vec_1: np.ndarray, vec_2: np.ndarray, third_axis: np.ndarray=None) -> float:
    """
    Return the angle between two vectors.
    """
    vec_1 = vec_1 / norm(vec_1)
    vec_2 = vec_2 / norm(vec_2)
    dot_prod = np.clip(np.dot(vec_1, vec_2), -1, 1)
    angle = np.arccos(dot_prod)
    if third_axis is None:
        return angle
    else:
        x_axis = np.cross(vec_1, third_axis)
        x = np.dot(vec_2, x_axis)
        y = np.dot(vec_2, vec_1)
        angle = np.arctan2(y, x)
        return angle


def path_length(vertices: np.ndarray=None, edges: np.ndarray=None, vec_list: list=None) -> float:
    """
    Get path length of a neurite branch.
    """
    if not edges:
        edges = []
    if not vec_list:
        vec_list = []
        for e in edges:
            p, c = e
            vec = vertices[c] - vertices[p]
            vec_list.append(vec)
    path_len = np.sum([norm(v) for v in vec_list])
    return path_len


def read_swc(file_path: str, flip_xz: bool=False) -> Skeleton:
    """
    Read .swc into a Skeleton object. If flip_xz, the coordinate will be converted from xyz into zyx (for numpy)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError('The SWC file was not found at: {}'.format(file_path))

    swc_str = []
    with open(file_path, 'r') as f:
        swc_lines = f.read().splitlines()

    for line in swc_lines:
        if line.startswith(' '):
            line = line[1:]
        swc_str.append(line)
    swc_str = '\n'.join(swc_str)
    skl = Skeleton.from_swc(swc_str)
    if flip_xz:
        skl.vertices = np.flip(skl.vertices, axis=1)
    return skl

def read_neuron(nrn_dir: str) -> None:
    """
    Read neuron from pickle file.
    """
    path = Path(nrn_dir)
    if not path.is_file():
        raise FileNotFoundError('The neuron file was not found at: {}'.format(nrn_dir))
    with open(nrn_dir, 'rb') as f:
        nrn = pickle.load(f)
    return nrn


def write_swc(swc_dir: str, swc: Union[Skeleton, str]) -> None:
    """
    Write the skeleton into a .swc file.
    """
    if isinstance(swc, Skeleton):
        swc_str = swc.to_swc()
    with open(swc_dir, 'w') as f:
        f.write(swc_str)


def rall_rule(n: float=30., r_p: float=1.0, split: float=1.0, sp_decay_coef: float=0.1) -> Tuple:
    """
    n is the power of rall's rule. the bigger the n, the smaller the decay after
    each branching. sp_decay_coef is the ratio between smaller split and large split.
    the bigger the value, the bigger the differentiation
    """
    split_decay = np.power(split, sp_decay_coef)
    coef = np.exp(-split_decay + 1)
    sum_child = 1 + np.power(coef, n)
    r_big = np.power(np.power(r_p, n) / sum_child, 1 / n)
    r_small = r_big * coef
    return r_small, r_big


def rot_basis(old_basis: dict, intrinsic: bool=True, order: str='zyz', alpha: float=0.0, beta: float=0.0,
              gamma: float=0.0) -> dict:
    """
    Rotate the basis with given Euler angles and rotation order under intrinsic or extrinsic setting.
    """
    new_basis = old_basis.copy()
    if intrinsic:
        for o, angle in zip(order, [alpha, beta, gamma]):
            for ax in [a for a in 'xyz' if a != o]:
                new_basis[ax] = axis_rot(new_basis[o], new_basis[ax], angle)
    elif order[-1] == 'z':
        for o, angle in zip(order[1:], [alpha, beta, gamma]):
            for ax in 'xyz':
                new_basis[ax] = axis_rot(old_basis[o], new_basis[ax], angle=angle)
    else:
        new_order = {'x': 'z', 'y': 'y', 'z': 'x'}
        for o, angle in zip(_branch_order, [alpha, beta, gamma]):
            for ax in 'xyz':
                new_basis[ax] = axis_rot(old_basis[new_order[o]], new_basis[ax], angle=angle)
    return new_basis


def get_euler_angles(old_basis: dict, new_basis: dict) -> float:
    """
    Get the Euler angles of two basis, which follows the zyz order.
    """
    old_basis_mat = np.vstack([old_basis[v] for v in 'xyz']).T
    new_basis_mat = np.vstack([new_basis[v] for v in 'xyz']).T
    rot_mat = new_basis_mat @ np.linalg.inv(old_basis_mat)
    rot = Rotation.from_matrix(rot_mat)
    euler_angles = rot.as_euler('XYZ')
    return euler_angles


def brownian_path(M: int, N: int) -> np.ndarray:
    """
    Get the brownian path, which is a stochastic process basis on Gaussian random walk.
    :param M: dimension
    :param N: numer of points
    """
    dt = 1.0 / N
    dt_sqrt = np.sqrt(dt)
    B = np.empty((M, N), dtype=np.float32)
    B[:, 0] = 0
    for n in range(N - 1):
        t = n * dt
        xi = np.random.randn(M) * dt_sqrt
        B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
    return B


def interp_spline(pts: np.ndarray, sample_num: int, loop: bool=True) -> np.ndarray:
    """
    Interp a spline to make the gap smoother. If the skeleton contains a loop, break it first.
    """
    x, y, z = pts.T
    tck, u = intp.splprep([x, y, z], s=0.0)
    if not loop:
        u_new = np.linspace(0, 1, sample_num)
        x_new, y_new, z_new = intp.splev(u_new, tck=tck)
        return np.array([x_new, y_new, z_new]).T
    else:
        dist_vec = [norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        normalized_x = np.cumsum(dist_vec) / np.sum(dist_vec)
        normalized_x = np.concatenate(([0], normalized_x))
        u1_new = np.linspace(normalized_x[1], normalized_x[-2], int(sample_num * (pts.shape[0] - 2) / pts.shape[0]))
        x1_new, y1_new, z1_new = intp.splev(u1_new, tck=tck)
        result_pt1 = np.array([x1_new, y1_new, z1_new]).T
        mid_pt = len(pts) // 2
        re_arr_pts = np.vstack([pts[mid_pt:-1], pts[:mid_pt], pts[mid_pt]])
        dist_vec2 = [norm(re_arr_pts[i + 1] - re_arr_pts[i]) for i in range(len(re_arr_pts) - 1)]
        normalized_x2 = np.cumsum(dist_vec2) / np.sum(dist_vec2)
        normalized_x2 = np.concatenate(([0], normalized_x2))
        x2, y2, z2 = re_arr_pts.T
        tck2, u2 = intp.splprep([x2, y2, z2], s=0.0)
        start = (len(pts) - 2 - mid_pt) % len(pts)
        u2_new = np.linspace(normalized_x2[start], normalized_x2[start + 2], int(sample_num * (2 / pts.shape[0])))
        x2_new, y2_new, z2_new = intp.splev(u2_new, tck=tck2)
        result_pt2 = np.array([x2_new, y2_new, z2_new]).T
        return np.vstack((result_pt1, result_pt2))


def single_neurite(vertices: np.ndarray=None, phi: Union[list, np.ndarray]=None, theta: Union[list, np.ndarray]=None,
                   step: Union[list, np.ndarray]=None, init_vec: np.ndarray=None, basis: dict=None,
                   return_basis: bool=False) -> Union[Skeleton, Tuple]:
    """
    Construct a single neurite branch based on angles between nodes or vertices list.
    """
    skl = init_skl()
    basis_list = []
    if vertices is None:
        if init_vec is None:
            init_vec = vec_from_sph(phi=phi[0], theta=theta[0], order='yzy')
        if basis is None:
            basis = init_basis(main_vec=init_vec, main_axis='y')
        skl.vertices = np.vstack([skl.vertices, np.array([0, 0, 0])])
        if theta is None:
            theta = np.ones_like(phi)
        if step is None:
            step = np.ones_like(phi)
        for p, t, s in zip(phi, theta, step):
            vec = vec_from_sph(phi=p, theta=t, basis=basis, order='yzy')
            new_pt = skl.vertices[-1] + vec * s
            skl.vertices = np.vstack([skl.vertices, new_pt])
            basis = rot_basis(old_basis=basis, order='yzy', intrinsic=False, beta=p, gamma=t)
            basis_list.append(basis)
    else:
        skl.vertices = vertices
    if skl.vertices.shape[0] == 1:
        skl.edges = np.zeros((0, 2))
    else:
        skl.edges = np.vstack([skl.edges, np.vstack([[i, i + 1] for i in range(len(skl.vertices) - 1)])])
    skl.radius = 0.2 * np.ones(len(skl.vertices))
    skl.vertex_types = np.ones(len(skl.vertices), dtype=int) * 2
    if return_basis:
        return skl, basis_list
    else:
        return skl


def combine_skl_no_conn(*skl: Skeleton) -> Skeleton:
    """
    Combine isolated skeletons without establishing connections.
    """
    gh = init_skl()
    for i, _skl in enumerate(skl):
        gh.edges = np.vstack([gh.edges, _skl.edges + gh.vertices.shape[0]])
        gh.vertices = np.vstack([gh.vertices, _skl.vertices])
        gh.vertex_types = np.concatenate((gh.vertex_types, _skl.vertex_types))
        gh.radius = np.concatenate((gh.radius, _skl.radius))
    gh.vertex_types = gh.vertex_types.astype(int)
    return gh


def cord2skl(basis: dict, scale: float=1, ctr: np.ndarray=None) -> Skeleton:
    """
    Convert a coordinate system into a tiny skeleton (for visualization).
    """
    cord = init_skl()
    if ctr is None:
        ctr = np.array([0, 0, 0])
    tp = {'x': 7, 'y': 8, 'z': 9}
    cord.vertex_types = np.array([6])
    cord.vertices = np.array([ctr])
    for t, d in basis.items():
        cord.vertices = np.vstack([cord.vertices, d * scale + ctr])
        cord.vertex_types = np.append(cord.vertex_types, tp[t])
    cord.edges = np.array([[0, 1], [0, 2], [0, 3]])
    cord.radius = np.array([0.1, 0.1, 0.1, 0.1])
    return cord


def align_basis(old_basis: dict, new_basis: dict) -> Tuple:
    """
    Get the Euler angles that will align two basis.
    ."""
    p, t = get_sph_angles(carte=new_basis['y'], basis=old_basis, order='yzy')
    temp_basis = rot_basis(old_basis=old_basis, intrinsic=False, alpha=0, beta=p, gamma=t, order='yzy')
    offset = get_angle(temp_basis['z'], new_basis['z'])
    if norm(axis_rot(u=temp_basis['y'], v=temp_basis['z'], angle=offset) - new_basis['z']) > _numeric_zero:
        offset = -offset
    return (p, t), offset


def find_internal_aux_vec(nrt_v: np.ndarray, reverse: bool=False) -> np.ndarray:
    """
    Find internal aux vec.
    """
    aux_v = np.array([1, 0, 0])
    if not reverse:
        init_v = get_unit_vec(nrt_v[1] - nrt_v[0])
        i = 2
        while i < len(nrt_v):
            aux_v = get_unit_vec(nrt_v[i] - nrt_v[i - 1])
            if norm(np.cross(aux_v, init_v)) > _numeric_zero:
                break
            i += 1
    else:
        init_v = get_unit_vec(nrt_v[-1] - nrt_v[-2])
        i = -2
        while -i < len(nrt_v) + 1:
            aux_v = get_unit_vec(nrt_v[i] - nrt_v[i - 1])
            if norm(np.cross(aux_v, init_v)) > _numeric_zero:
                break
            i -= 1
    return aux_v
