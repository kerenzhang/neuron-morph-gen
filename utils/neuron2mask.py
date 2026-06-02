from typing import List, Tuple
import numpy as np
from typing import Union
from cloudvolume import Skeleton
from scipy import spatial
from skimage.draw import polygon2mask
from scipy.spatial import KDTree

from utils.utils import *
from components.neuron import Neuron
from components.neurite import Axon, Neurite

_collapse_neurite = True
_soma_type = 1


def get_bbox_neurite(skl: Union[Skeleton, list]) -> Tuple:
    """
    return coordinates of bounding box of skeletons, with radius concerned
    """
    if isinstance(skl, Skeleton):
        nodes = skl.vertices
        radius = skl.radius
        min_x, min_y, min_z = np.min(nodes - np.tile(radius, (3, 1)).transpose(), axis=0)
        max_x, max_y, max_z = np.max(nodes + np.tile(radius, (3, 1)).transpose(), axis=0)
        return (min_x, max_x), (min_y, max_y), (min_z, max_z)
    elif isinstance(skl, list) and len(skl) == 1:
        return get_bbox_neurite(skl[0].skl)
    else:
        bboxes = []
        for ind_skl in skl:
            if isinstance(ind_skl, Axon):
                ind_skl = ind_skl.skl
            ind_bbox = get_bbox_neurite(ind_skl)
            bboxes.append(ind_bbox)
        mins = np.vstack([np.hstack(b) for b in bboxes])[:, 0::2]
        maxs = np.vstack([np.hstack(b) for b in bboxes])[:, 1::2]
        min_x, min_y, min_z = np.min(mins, axis=0)
        max_x, max_y, max_z = np.max(maxs, axis=0)
        return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def calculate_offset(bbox_neurite: tuple, bbox_soma: tuple, image_shape: List=None, min_pad: Tuple=(0.1, 0.2, 0.2)) -> Tuple:
    """
    Calculate offset needs to be added to the skl vertices and soma points given an image shape, assuming
    the center of bounding box is in the image center. The bbox of soma and neurite is overlapped to determine
    the final image mask shape and offset. If no shape is given, it will generate a minimum image with padding.

    Because the neurite will be translated to the edge of the soma, an average of bbox shape of soma is added
    to the padding.
    """
    bbox_soma, bbox_neurite = (np.array(bbox_soma), np.array(bbox_neurite))

    min_x, min_y, min_z = [min(a, b) for a, b in zip(bbox_neurite[:, 0], bbox_soma[:, 0])]
    max_x, max_y, max_z = [max(a, b) for a, b in zip(bbox_neurite[:, 1], bbox_soma[:, 1])]
    bbox = ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    bbox_shape = [b[1] - b[0] for b in bbox]
    bbox_center = [(b[1] + b[0]) / 2 for b in bbox]

    min_shape = []
    for box_axis, pad_perc in zip(bbox_shape, min_pad):
        axis_pad = np.ceil(box_axis * pad_perc)
        axis_len = np.ceil(box_axis) + 2 * axis_pad + np.max([b[1] - b[0] for b in bbox_soma])
        min_shape.append(int(axis_len))

    if image_shape is None:
        image_shape = min_shape
    else:
        for i in range(3):
            image_shape[i] = max(image_shape[i], min_shape[i])

    offset = [image_shape[i] / 2 - bbox_center[i] for i in range(3)]

    return np.array(offset), image_shape


class Node:

    def __init__(self, pt: Union[Tuple, np.ndarray], r: float, type_id: int) -> None:
        self.c = pt
        self.r = r
        self.id = type_id


class Segment:
    """
    Segment that contains two connected nodes
    """
    def __init__(self, s: Node, t: Node) -> None:
        self.s = s
        self.t = t
        self.seg_v = self.t.c - self.s.c
        self.len_2 = np.dot(self.seg_v, self.seg_v)
        self.len = np.sqrt(self.len_2)

    def check_type(self) -> int:
        if self.s.id == self.t.id and self.s.id == _soma_type:
            return 0
        elif self.s.id == self.t.id and self.s.id != _soma_type:
            return 1
        else:
            return 0

    def get_bbox_pts(self) -> np.ndarray:
        min_x, min_y, min_z = [np.floor(min(self.s.c[i] - self.s.r, self.t.c[i] - self.t.r)) for i in range(3)]
        max_x, max_y, max_z = [np.ceil(max(self.s.c[i] + self.s.r, self.t.c[i] + self.t.r)) for i in range(3)]
        pts_mesh_grid = np.array(np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1), np.arange(min_z, max_z + 1))).T.reshape(-1, 3)
        return pts_mesh_grid.astype(int)

    def judge_bbox(self, mask_img: np.ndarray) -> np.ndarray:
        """
        Assign voxel type for all voxels based on structure filling.
        """
        pts = self.get_bbox_pts()
        results = assign_bbox(pts, self)
        for p, value in zip(pts, results):
            if mask_img[tuple(p)] == 0:
                mask_img[tuple(p)] = value * self.t.id
        return mask_img


class soma_entity:

    def __init__(self, pts: np.ndarray) -> None:
        self.pts = pts
        if self.pts.shape[0] < 5:
            self.pts = np.ones((3, 3))
        self.poly_pts = np.zeros((0, 3))
        min_x, min_y, min_z = np.min(self.pts, axis=0)
        max_x, max_y, max_z = np.max(self.pts, axis=0)
        self.bbox = ((min_x, max_x), (min_y, max_y), (min_z, max_z))

    def judge_bbox(self, offset: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Polygon filling with respect to tetrahedron.
        """
        pts_offset = self.pts + offset
        bbox_offset = np.array(self.bbox) + np.vstack((offset, offset)).T
        if pts_offset.shape[0] < 5:
            return mask

        hull = spatial.Delaunay(pts_offset)
        tetra_simp = hull.simplices.copy()
        tetra_order = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        z_list = np.arange(int(np.ceil(bbox_offset[2][0])), int(np.floor(bbox_offset[2][1])) + 1)
        for z in z_list:
            pts_z = np.ones((0, 3))
            for edges in tetra_simp:
                for order in tetra_order:
                    p = pts_offset[edges[order[0]]]
                    t = pts_offset[edges[order[1]]]
                    intcpt = get_intercept(p, t, ppt=np.array([0, 0, z]))
                    if intcpt.shape[0] > 0:
                        pts_z = np.vstack((pts_z, intcpt))

            pts_z = np.unique(pts_z, axis=0)
            pts_z = filter_close_points_kdtree(pts_z, 3.0)
            if pts_z.shape[0] > 5:
                mask_2D = criteria_polygon_2D(vertices=pts_z, mask_shape=mask.shape[0:2], i=z)
                mask[:, :, z] = mask_2D
                self.poly_pts = np.vstack((self.poly_pts, pts_z))

        return mask

def filter_artifacts(mask: np.ndarray) -> np.ndarray:
    """Filter artifacts."""
    pass


def criteria_sphere(pt: np.ndarray, seg: Segment) -> bool:
    """
    Check if a point belongs within a sphere.
    """
    dist_s = pt - seg.s.c
    dist_t = pt - seg.t.c
    value = False
    if np.dot(dist_s, dist_s) <= np.square(seg.s.r) or np.dot(dist_t, dist_t) <= np.square(seg.t.r):
        value = True
    return value

def criteria_cylinder(pt: np.ndarray, seg: Segment) -> int:
    """
    Check if a point belongs within a cylinder.
    """
    dist_v = pt - seg.s.c
    cross_dot = np.dot(dist_v, seg.seg_v)
    if cross_dot > 0:
        if cross_dot < seg.len_2:
            prx = cross_dot / seg.len
            frx = prx / seg.len
            rad = frx * np.clip(seg.t.r, 1, 100) + (1 - frx) * np.clip(seg.s.r, 1.0, 100)
            if np.dot(dist_v, dist_v) - np.square(prx) <= np.square(rad):
                return 1
        elif np.dot(pt - seg.t.c, pt - seg.t.c) <= np.square(seg.t.r):
            return 0

    return 0


def criteria_polygon_2D(vertices: np.ndarray, mask_shape: Union[Tuple, List]) -> np.ndarray:
    """
    Check if a point belongs within 2D polygon.
    """
    hull = spatial.ConvexHull(vertices[:, 0:2])
    hull_v = hull.vertices
    hull_v = np.append(hull_v, hull_v[0])
    hull_v = vertices[hull_v]
    intp_pts = interp_spline(pts=hull_v, sample_num=len(hull_v) * 3)
    mask_2D = polygon2mask(image_shape=mask_shape, polygon=intp_pts[:, 0:2])
    return mask_2D


def get_intercept(e1: np.ndarray, e2: np.ndarray, ppt: np.ndarray=np.array([0, 0, 0]),
                  nv: np.ndarray=np.array([0, 0, 1])) -> np.ndarray:
    """
    Return the interception of a point and a line.
    """
    r1 = proj(nv, e1 - ppt)
    r2 = proj(nv, e2 - ppt)
    if np.dot(r1, r2) < 0:
        vec = e2 - e1
        t = np.dot(ppt - e1, nv) / np.dot(vec, nv)
        return t * vec + e1
    elif np.dot(r1, r2) == 0:
        r1_pt = np.zeros((0, 3))
        r2_pt = np.zeros((0, 3))
        if np.linalg.norm(r1) == 0:
            r1_pt = e1
        if np.linalg.norm(r2) == 0:
            r2_pt = e2
        return np.vstack((r1_pt, r2_pt))
    else:
        return np.zeros((0, 3))


def assign_bbox(pts: List[np.ndarray], seg: Segment) -> list:
    bool_vec = []
    for i, pt in enumerate(pts):
        if criteria_cylinder(pt, seg) or criteria_sphere(pt, seg):
            bool_vec.append(True)
        else:
            bool_vec.append(False)
    return bool_vec


def filter_close_points_kdtree(points: np.ndarray, threshold: float) -> np.ndarray:
    points_array = np.array(points)
    tree = KDTree(points_array)
    pairs = tree.query_pairs(threshold)

    indices_to_remove = set()
    for i, j in pairs:
        indices_to_remove.add(j)
    filtered_points = [point for i, point in enumerate(points) if i not in indices_to_remove]

    return np.array(filtered_points)


def swc2mask(skl: Union[Skeleton, Axon, Neurite], offset: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    convert skeletons in .swc representation to mask
    """

    if isinstance(skl, Axon) or isinstance(skl, Neurite):
        skl = skl.skl

    vts_offset = skl.vertices.copy() + offset
    radius = skl.radius.copy()
    vertex_types = skl.vertex_types.copy()

    edges = skl.edges.copy()
    for e in edges:
        s_index, t_index = e[:]
        s_node = Node(vts_offset[s_index], radius[s_index], vertex_types[s_index])
        t_node = Node(vts_offset[t_index], radius[t_index], vertex_types[t_index])
        seg = Segment(s_node, t_node)
        if seg.check_type():
            seg.judge_bbox(mask)

    return mask


def neuron2mask(nrn: Neuron, ori_shape: Union[Tuple, List]=None, return_bbox: bool=False) -> Union[Tuple, np.ndarray]:
    """
    Convert grown neuron to mask
    """
    neurite_id, all_neurite = (list(nrn.all_skl.keys()), list(nrn.all_skl.values()))
    bbox_neurite = get_bbox_neurite(all_neurite)
    soma = nrn.elements['soma'][-1]
    soma_instance = soma_entity(soma.vertices)

    offset, mask_shape = calculate_offset(bbox_neurite, soma_instance.bbox, ori_shape)
    img_mask = np.zeros(mask_shape, dtype=np.uint8)
    img_mask = soma_instance.judge_bbox(offset=offset, mask=img_mask)
    img_mask[img_mask != 0] = _soma_type

    for i, s in zip(neurite_id, all_neurite):
        stem_vec = nrn.elements['neurite'][i][0].init_vec
        stem_pt = nrn.soma_stem[i]
        stem_offset = soma.get_stem_pts(stem_vec) - stem_pt + offset
        img_mask = swc2mask(s, offset, img_mask)

    if _collapse_neurite:
        img_mask[img_mask > _soma_type] = _soma_type + 1

    img_mask_cls = np.zeros_like(img_mask)
    img_mask_cls[img_mask == 1] = 2
    img_mask_cls[img_mask == 2] = 1

    if not return_bbox:
        return img_mask_cls.T
    else:
        return img_mask_cls.T, offset


def mix_neuron_mask(*ind_info: np.ndarray) -> np.ndarray:
    """
    Merge multiple neurons into a single mask
    # TODO expand to multiple neurons
    """

    n1 = ind_info[0]
    n2 = ind_info[1]
    img_1 = n1[0].T
    img_2 = n2[0].T
    delta = np.array(n1[1]) - np.array(n2[1])
    pad_1, pad_2 = ([], [])
    for d in delta:
        pad_1.append([np.ceil(np.max([0, -d])).astype(int), 0])
        pad_2.append([np.ceil(np.max([0, d])).astype(int), 0])
    img_1 = np.pad(img_1, pad_1, mode='constant', constant_values=0)
    img_2 = np.pad(img_2, pad_2, mode='constant', constant_values=0)
    new_mask_shape = [np.max([img_1.shape[i], img_2.shape[i]]) for i in range(3)]

    pad_1, pad_2 = ([], [])
    for i, ns in enumerate(new_mask_shape):
        pad_1.append([0, ns - img_1.shape[i]])
        pad_2.append([0, ns - img_2.shape[i]])

    img_1 = np.pad(img_1, pad_1, mode='constant', constant_values=0)
    img_2 = np.pad(img_2, pad_2, mode='constant', constant_values=0)
    new_img = np.zeros(new_mask_shape, dtype=np.uint8)

    for j in range(1, 3):
        mix = np.logical_or(img_1 == j, img_2 == j)
        new_img[mix] = j

    return new_img.T


