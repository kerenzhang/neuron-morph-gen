import numpy as np
from utils.params_var import get_dist
from components.base_classes import Node, Element
from utils.utils import *


class SOMA(Element):
    """
    Construct a soma (cell body) with volume, shape, and noise. Return vertices in 3D representing mesh surface points.
    Currently one has sphere and ellipsoid options
    """
    def __init__(self, volume: float=10000.0, shape: str='sphere', res: int=10, noise: float=0.3) -> None:
        """
        :param volume: volume of soma in micron^3
        :param shape: shape type
        :param res: resolution of vertices points (distance between surface vertices)
        :param noise: noise level of the shape construction
        """
        super(SOMA, self).__init__()
        self.contact_nodes = []
        self.vertices = np.zeros((0, 3))
        self.ctr = np.array([0, 0, 0])
        self.skl_r = np.power(volume / (np.pi * 4 / 3), 1 / 3)

        if shape == 'sphere':
            self.vertices = self.sphere_soma(volume=volume, res=res, noise=noise)
        elif shape == 'ellipsoid':
            self.vertices = self.ellipsoid_soma(volume=volume, res=res, noise=noise)
        else:
            print('shape must be sphere or ellipsoid')

    @staticmethod
    def sphere_soma(volume: float, res: float=10, noise: float=0.3) -> np.ndarray:
        max_r = np.power(volume / (np.pi * 4 / 3), 1 / 3)
        max_num_pts = np.ceil(2 * np.pi * max_r / res).astype(int)

        angle_res = 2 * np.pi / max_num_pts
        angle_0 = np.arange(max_num_pts) + get_dist('uniform', max_num_pts, 0, noise)
        angle_0 *= angle_res

        # add noise in form of brownian walk
        shape_deform = brownian_path(1, len(angle_0))[0] * noise
        r_noise_0 = max_r * (1 + shape_deform)
        pt_list = np.vstack([[np.cos(a) * r, np.sin(a) * r, 0] for a, r in zip(angle_0, r_noise_0)])
        z_num = np.ceil(max_num_pts / 4).astype(int) + 1
        z_res = 0.5 * np.pi / z_num

        # edge case, if only 2 z, compress to 1
        if z_num == 2:
            z_num = 1

        # upper half
        angle_prv = np.array([angle_0])
        for z in np.arange(1, z_num):
            r_z = np.cos(z * z_res) * r_noise_0
            r_noise_z = np.squeeze(np.clip(r_z * get_dist('uniform', max_num_pts, 1, noise), 0, 9999))
            angle_z = np.squeeze(angle_prv[-1] * get_dist('uniform', max_num_pts, 1, noise))
            pts_z = np.array([[np.cos(a) * r, np.sin(a) * r, np.sin(z * z_res) * max_r] for a, r in zip(angle_z, r_noise_z)])
            pt_list = np.vstack((pt_list, pts_z))
            angle_prv = np.vstack((angle_prv, angle_z))

        # lower half
        angle_prv = np.array([angle_0])
        for z in np.arange(-1, -z_num, -1):
            r_z = np.cos(z * z_res) * r_noise_0
            r_noise_z = np.squeeze(np.clip(r_z * get_dist('uniform', max_num_pts, 1, noise), 0, 9999))
            angle_z = np.squeeze(angle_prv[-1] * get_dist('uniform', max_num_pts, 1, 0))
            pts_z = [[np.cos(a) * r, np.sin(a) * r, np.sin(z * z_res) * max_r] for a, r in zip(angle_z, r_noise_z)]
            pt_list = np.vstack((pt_list, pts_z))
            angle_prv = np.vstack((angle_prv, angle_z))
        pt_list = np.vstack((pt_list, np.array([0, 0, max_r]), np.array([0, 0, -max_r])))

        return pt_list

    @staticmethod
    def ellipsoid_soma(volume: float, ecn_1: float=0.5, ecn_2: float=0.5, res: float=10, noise: float=0.0) -> np.ndarray:

        a = np.power(volume / (ecn_1 * ecn_2 * np.pi * 4 / 3), 1 / 3)
        b = a * ecn_1
        c = a * ecn_2

        max_num_pts = np.ceil(2 * np.pi * a / res).astype(int)
        angle_res = 2 * np.pi / max_num_pts
        angle_0 = np.arange(max_num_pts) + get_dist('uniform', max_num_pts, 0, noise)
        angle_0 *= angle_res

        noise = brownian_path(1, len(angle_0)) * noise
        a_noise_0 = a * (1 + noise[0])
        b_noise_0 = b * (1 + noise[0])
        pt_list = np.vstack([[np.cos(theta) * a, np.sin(theta) * b, 0] for theta, a, b in zip(angle_0, a_noise_0, b_noise_0)])
        z_num = np.ceil(max_num_pts / 2).astype(int) + 1
        z_res = 0.5 * np.pi / z_num

        print('ellipsoid soma: r is {}, max num is {}, z num is {}'.format(a, max_num_pts, z_num))

        angle_prv = np.array([angle_0])
        for z in np.arange(1, z_num):
            a_z = np.cos(z * z_res) * a_noise_0
            b_z = np.cos(z * z_res) * b_noise_0
            a_noise_z = np.squeeze(np.clip(a_z * get_dist('norm', max_num_pts, 1, noise), 0, 9999))
            b_noise_z = np.squeeze(np.clip(b_z * get_dist('norm', max_num_pts, 1, noise), 0, 9999))
            angle_z = np.squeeze(angle_prv[-1] * get_dist('uniform', max_num_pts, 1, noise))
            pts_z = np.array([[np.cos(theta) * a, np.sin(theta) * b, np.sin(z * z_res) * c] for theta, a, b in zip(angle_z, a_noise_z, b_noise_z)])
            pt_list = np.vstack((pt_list, pts_z))
            angle_prv = np.vstack((angle_prv, angle_z))

        angle_prv = np.array([angle_0])
        for z in np.arange(-1, -z_num, -1):
            a_z = np.cos(z * z_res) * a_noise_0
            b_z = np.cos(z * z_res) * b_noise_0
            a_noise_z = np.squeeze(np.clip(a_z * get_dist('uniform', max_num_pts, 1.0, noise), 0, 9999))
            b_noise_z = np.squeeze(np.clip(b_z * get_dist('uniform', max_num_pts, 1.0, noise), 0, 9999))
            angle_z = np.squeeze(angle_prv[-1] * get_dist('uniform', max_num_pts, 1.0, noise))
            pts_z = np.array([[np.cos(theta) * a, np.sin(theta) * b, np.sin(z * z_res) * a * ecn_2] for theta, a, b in zip(angle_z, a_noise_z, b_noise_z)])
            pt_list = np.vstack((pt_list, pts_z))
            angle_prv = np.vstack((angle_prv, angle_z))
        pt_list = np.vstack((pt_list, np.array([0, 0, c]), np.array([0, 0, -c])))

        return pt_list

    def get_stem_pts(self, init_vec: np.ndarray) -> np.ndarray:
        """
        Return the stem points connecting to the axons closest vertices to the neurite starting position.
        """
        center_pts = np.mean(self.vertices, axis=0)
        stem_pts = np.zeros((0, 3))
        ray = np.vstack([(v - center_pts) / np.linalg.norm(v - center_pts) for v in self.vertices])

        if init_vec.ndim == 1:
            init_vec = np.array([init_vec])

        for stem_vec in init_vec:
            phi, theta = get_sph_angles(stem_vec, order='zyz')
            new_vec = vec_from_sph(phi=phi, theta=theta)
            dist = np.dot(ray, new_vec)
            stem_pt = self.vertices[np.argmax(dist)]
            stem_pts = np.vstack((stem_pts, stem_pt))

        return stem_pts

