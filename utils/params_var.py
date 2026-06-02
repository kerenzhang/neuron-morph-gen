from typing import Any, List, Tuple
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass
from utils.utils import *
import scipy.stats as stats
from scipy.optimize import curve_fit
__all__ = ['mv_param', 'global_params', 'get_dist', 'max_order', 'num_segments', 'init_r',
           'init_vec', 'symmetry', 'd2h_ratio', 'length', 'phi', 'step_size', 'theta', 'ampl', 'roll', 'tilt', 'ratio',
           'expand_coef', 'hist_w', 'contr_factor', 'proj_w', 'num_axons', 'num_apical_dendrites', 'num_basal_dendrites',
           'neuron_params', 'interp']

class global_params:
    expand_coef = 0.85
    rall = 4.0
    central = 1.0


@dataclass(kw_only=True)
class mv_param:
    """
    parameters that follow the mean and variance distribution models
    """
    name: str = 'param'
    values: tuple = (0.0, 0.0)
    belonging: str = 'segments'
    dist_type: str = 'lognorm'
    scope: str = 'neurite'
    scaler: bool = True
    description: str = 'general class for mean and variance distribution models'

    def __init__(self, **kwargs) -> None:
        """Initialize the object."""
        if 'values' in kwargs:
            self.values = kwargs['values']

    def sample(self, num: int) -> np.ndarray:
        """
        sample given distribution. if num == 1 it will return a single number.
        """
        results = get_dist(self.dist_type, num, *self.values)
        return results

    def fit(self, data: np.ndarray) -> Tuple:
        """
        fit distribution with respect to given distribution type
        """
        return fit_dist(data, self.dist_type)

    @classmethod
    def collect_from_gui(cls, binder: Any) -> List:
        """
        Handles data input from GUI
        """
        result_list = []
        bd_type, bd_order, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'mean', 'var']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            if cls.dist_type == 'lognorm':
                values = (np.log(float(bd_mean[i])), float(bd_var[i]))
            else:
                values = (float(bd_mean[i]), float(bd_var[i]))
            result_list.append({name: values})
        return result_list


@dataclass(kw_only=True)
class ctrl_param:
    name: str = 'param'
    values: tuple = (0., 0., 0.)


class spl_ctrl_pts(ctrl_param):
    name: str = 'spl_ctrl_pts'
    values: tuple = (0.0, 0.0, 0)


class reverse(ctrl_param):
    name: str = 'reverse'
    values: bool = False


@dataclass(kw_only=True)
class choice_param:
    """
    Choice type parameters
    """
    name: str = 'param'
    TF_values: bool = False
    values: str = ''

    def __init__(self, **kwargs) -> None:
        if 'values' in kwargs:
            self.values = kwargs['values']

    @classmethod
    def collect_from_gui(cls, binder: Any) -> List:
        """
        Handles data input from GUI
        """
        result_list = []
        bd_type, bd_order, bd_choice = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'choice']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            values = str(bd_choice[i])
            result_list.append({name: values})
        return result_list


###############################################################################
class step_size(mv_param):
    name = 'step_size'
    dist_type = 'norm'
    values = (3, 0.5)
    scope = 'node'
    scaler = True
    description = 'stride step size between individual nodes'

class phi(mv_param):
    name = 'phi'
    dist_type = 'lognorm'
    values = (-0.8, 0.1, -0.0)
    scope = 'node'
    scaler = True
    description = 'magnitude of angle change w.r.t. previous vector (elevation)'

class theta(mv_param):
    name = 'theta'
    dist_type = 'bi-vonmises'
    values = (0.55, 0, 3, 3.14, 3)
    scope = 'node'
    scaler = True
    description = 'rotation of the new vector (azimuth)'

    @classmethod
    def collect_from_gui(cls, binder: Any) -> List:
        """Collect from gui."""
        result_list = []
        bd_type, bd_order, bd_ratio, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'ratio', 'mean', 'var']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            _ratio = float(bd_ratio[i])
            values = (_ratio, float(bd_mean[i][0]), float(bd_var[i][0]), float(bd_mean[i][1]), float(bd_var[i][1]))
            result_list.append({name: values})
        return result_list

class length(mv_param):
    name = 'length'
    dist_type = 'bi-lognorm'
    values = (0.685, 3.13, 0.838, 5.11, 0.324)
    scope = 'neurite'
    scaler = True
    description = 'length of individual branch (all nodes between bifurcations or terminus'

    def sample(self, num: int) -> np.float64:
        """Sample."""
        results = super().sample(num)
        return np.clip(results, 0.001, 300)

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        """Collect from gui."""
        result_list = []
        bd_type, bd_order, bd_ratio, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'ratio', 'mean', 'var']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            _ratio = float(bd_ratio[i])
            values = (_ratio, np.log(float(bd_mean[i][0])), float(bd_var[i][0]), np.log(float(bd_mean[i][1])), float(bd_var[i][1]))
            result_list.append({name: values})
        return result_list


class tilt(mv_param):
    name = 'tilt'
    dist_type = 'lognorm'
    values = (np.log(0.2), 0.05, 0.0)
    scope = 'neurite'
    scaler = True
    description = 'tilt angle of the child plane w.r.t the parent plane'

    def sample(self, num: int) -> np.ndarray:
        """Sample."""
        results = super().sample(num)
        return np.clip(results, -np.pi / 2, np.pi / 6)


class roll(mv_param):
    name = 'roll'
    dist_type = 'vonmises'
    values = (1.5, 2.5)
    scope = 'neurite'
    scaler = True
    description = 'roll angle of the child plane w.r.t the parent plane'

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        """
        Handles data input from GUI
        """
        result_list = []
        bd_type, bd_order, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'mean', 'var']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            values = (float(bd_mean[i]), float(bd_var[i]))
            result_list.append({name: values})
        return result_list


class ampl(mv_param):
    name = 'ampl'
    dist_type = 'lognorm'
    values = (np.log(1.5), 0.5, 0)
    scope = 'neurite'
    scaler = True
    description = 'amplitude of the bifurcation angle between two children vectors'

    def sample(self, num: int) -> np.float64:
        results = super().sample(num)
        return np.clip(results, 0.1, np.pi * 1.2)


class ratio(mv_param):
    name = 'ratio'
    dist_type = 'exponential-uniform'
    values = (0.5, 1.04, 1.05, 0, 55)
    scope = 'neurite'
    scaler = True
    description = 'ratio of the angle of the two children vectors w.r.t the parent vector. Always the bigger over smaller'

    def sample(self, num: int) -> int:
        results = super().sample(num)
        return np.clip(results, 1, 1e7)

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        result_list = []
        bd_type, bd_order, bd_ratio, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'ratio', 'mean', 'var']]
        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            _ratio = float(bd_ratio[i])
            values = (_ratio, float(bd_mean[i][0]), float(bd_var[i][0]), float(bd_mean[i][1]), float(bd_var[i][1]))
            result_list.append({name: values})
        return result_list


class expand_coef(mv_param):
    name = 'expand_coef'
    dist_type = 'norm'
    values = (1, 0.1)
    scope = 'axon'
    scaler = True
    description = 'The degree of expansion of the branch compared to the bifurcation angle'

    def sample(self, num: int) -> np.float64:
        results = super().sample(num)
        return np.clip(results, 0, 2)


class init_vec(mv_param):
    name = 'init_vec'
    dist_type = 'vonmises_fisher'
    values = ((1.57, 1.57), 1.5)
    scope = 'axon'
    scaler = False
    description = 'direction of the stemming vector out of soma'

    def sample(self, num: int) -> np.ndarray:
        results = super().sample(num)
        if num == 1:
            results = results[0]
        return results

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        result_list = []
        bd_type, bd_order, bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['type', 'order', 'mean', 'var']]

        for i, _t, _o in zip(range(len(bd_type)), bd_type, bd_order):
            name = '{}-{}'.format(int(_t), int(_o))
            x, y, z = [float(k) for k in bd_mean[i]]
            _phi, _theta = get_sph_angles(carte=np.array([x, y, z]), basis=_default_basis)
            values = ((_phi, _theta), float(bd_var[i]))
            result_list.append({name: values})

        return result_list


class init_r(mv_param):
    name = 'init_r'
    dist_type = 'lognorm'
    values = (np.log(1.1), 0.8)
    scope = 'axon'
    scaler = True
    description = 'the radius of the stemming nodes'

    def sample(self, num: int) -> float:
        results = super().sample(num)
        return np.clip(results, 0.1, 1e7)


class hist_w(mv_param):
    name = 'hist_w'
    dist_type = 'norm'
    values = (0.6, 0.1)
    scope = 'axon'
    scaler = True
    description = 'weight of history vectors which "smooth" the neurite in an "moving average window"'


class proj_w(mv_param):
    name = 'proj_w'
    dist_type = 'norm'
    values = (3, 0.5)
    scope = 'axon'
    scaler = True
    description = 'weight of bifurcation/init vectors which guide the growth towards a specific direction and prevents large deviation'


class contr_factor(mv_param):
    name = 'contr_factor'
    dist_type = 'norm'
    values = (0.95, 0.05)
    scope = 'axon'
    scaler = True
    description = 'contribution of the radius along the neurite'

    def sample(self, num: int) -> float:
        results = super().sample(num)
        return np.clip(results, 0, 100)


class max_order(mv_param):
    name = 'max_order'
    dist_type = 'norm'
    values = (5, 1)
    scope = 'axon'
    scaler = True
    description = 'maximum order of the axon tree (stemming node as the root)'

    def sample(self, num: int) -> int:
        results = super().sample(num)
        return np.round(np.clip(results, 1, 1e3)).astype(int)


class num_segments(mv_param):
    name = 'num_segments'
    dist_type = 'norm'
    values = (9, 3)
    scope = 'axon'
    scaler = True

    def sample(self, num: int) -> int:
        results = super().sample(num)
        return np.round(np.clip(results, 1, 1e5)).astype(int)


class symmetry(mv_param):
    name = 'symmetry'
    dist_type = 'uniform'
    values = (0.5, 0.1)
    scope = 'axon'
    scaler = True

    def sample(self, num: int) -> np.ndarray:
        results = super().sample(num)
        return np.clip(results, 0, 1.0)


class d2h_ratio(mv_param):
    name = 'd2h_ratio'
    dist_type = 'uniform'
    values = (1.3, 0.1)
    scope = 'axon'
    scaler = True

    def sample(self, num: int) -> float:
        results = super().sample(num)
        return np.clip(results, 1.0, 2.0)


class num_axons(mv_param):
    name = 'num_axons'
    values = (1, 0.001)
    dist_type = 'uniform'
    scope = 'neuron'
    scaler = True

    def sample(self, num: int) -> float:
        results = super().sample(num)
        return np.clip(np.round(results), 0, 1e3).astype(int)

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        result_list = []
        bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['mean', 'var']]
        name = '1-1'
        for m, v in zip(bd_mean, bd_var):
            result_list.append({name: (float(m), float(v))})
        return result_list


class num_apical_dendrites(mv_param):
    name = 'num_apical_dendrites'
    values = (1, 0.001)
    dist_type = 'uniform'
    scope = 'neuron'
    scaler = True

    def sample(self, num: int) -> int:
        results = super().sample(num)
        return np.clip(np.round(results), 0, 1e6).astype(int)

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        result_list = []
        bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['mean', 'var']]
        name = '1-1'
        for m, v in zip(bd_mean, bd_var):
            result_list.append({name: (float(m), float(v))})
        return result_list


class num_basal_dendrites(mv_param):
    name = 'num_basal_dendrites'
    values = (1, 0.001)
    dist_type = 'uniform'
    scope = 'neuron'
    scaler = True

    def sample(self, num: int) -> int:
        results = super().sample(num)
        return np.clip(np.round(results), 0, 1e6).astype(int)

    @classmethod
    def collect_from_gui(cls, binder: dict) -> List:
        """Collect from gui."""
        result_list = []
        bd_mean, bd_var = [list(getattr(binder, attr).values()) for attr in ['mean', 'var']]
        name = '1-1'
        for m, v in zip(bd_mean, bd_var):
            result_list.append({name: (float(m), float(v))})
        return result_list


class order_scheme(choice_param):
    name = 'order_scheme'
    values = 'c'


class basic_params:
    """
    Base class for all parameters in the template.
    """
    def __init__(self, **kwargs) -> None:
        self.attrs = {}
        self.params = {}

    def update_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.attrs.keys():
                self.attrs.update({key: value})
            elif key in self.params.keys():
                self.params.update({key: value})
            else:
                print('param {} not defined'.format(key))

    def to_dict(self) -> dict:
        basic_dict = {}
        for key, value in self.attrs.items():
            basic_dict.update({key: value})
        for key, value in self.params.items():
            basic_dict.update({key: value.values})
        return basic_dict

    def __repr__(self) -> str:
        verbose = []
        for key, value in self.attrs.items():
            verbose.append('{}: {}'.format(key, value))
        for key, value in self.params.items():
            verbose.append('{}: {}'.format(key, value.values))
        return 'params: ' + ', '.join(verbose)


class branch_params(basic_params):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.params = {'init_vec': init_vec, 'length': length, 'step_size': step_size, 'contr_factor': contr_factor,
                       'phi': phi, 'theta': theta, 'hist_w': hist_w, 'proj_w': proj_w}
        self.attrs = {'id': 0, 'type_id': 2, 'order': 1, 'from_soma': 0, 'start': [0, 0, 0]}
        self.update_params(**kwargs)
        if self.attrs['order'] == 1:
            self.attrs.update({'from_soma': 1})

    def __repr__(self) -> str:
        verbose = super().__repr__()
        return 'Neurite ' + verbose


class bif_params(basic_params):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.params = {'tilt': tilt, 'roll': roll, 'ampl': ampl, 'ratio': ratio, 'expand_coef': expand_coef}
        self.attrs = {'order': 0, 'type_id': 2}
        self.update_params(**kwargs)

    def __repr__(self) -> str:
        verbose = super().__repr__()
        return 'Bifurcation ' + verbose


class axon_params(basic_params):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.neurite = {}
        self.branches = {}
        self.attrs = {'id': 0, 'type_id': 2}
        self.params = {'init_vec': init_vec, 'max_order': max_order, 'init_r': init_r, 'num_branches': num_segments}
        self.update_params(**kwargs)
        self.branch_params = {0: branch_params()}
        self.bifurcations = {0: bif_params()}

    def add_neurite(self, **kwargs) -> None:
        """
        Add specific degree of neurite with num, not single neurite.
        """
        neurite = branch_params(**kwargs)
        order = neurite.attrs['order']
        neurite.attrs['type_id'] = self.attrs['type_id']
        self.neurite.update({order: neurite})

    def add_bif(self, **kwargs) -> None:
        """Add the bif."""
        bi = bif_params(**kwargs)
        self.bifurcations.update({bi.attrs['order']: bi})

    def to_dict(self) -> dict:
        axon_dict = super().to_dict()
        for i, neurite in self.neurite.items():
            axon_dict['neurite'] = {i: neurite.to_dict()}
        for j, bifurcation in self.bifurcations.items():
            axon_dict['bifurcation'] = {j: bifurcation.to_dict()}
        return axon_dict


class neuron_params:

    def __init__(self, template: str=None) -> None:
        self.axons = {}
        self.params = {}
        self.choices = {}
        self.template = template
        __currently_defined_params = ['max_order', 'num_segments', 'symmetry', 'd2h_ratio', 'init_r', 'init_vec', 'length',
                                      'hist_w', 'proj_w', 'contr_factor', 'phi', 'theta', 'step_size', 'ampl',
                                      'tilt', 'ratio', 'roll', 'num_apical_dendrites', 'num_basal_dendrites', 'num_axons']
        __currently_defined_choices = ['order_scheme']

        for __p in __currently_defined_params:
            p_class = globals()[__p]()
            if isinstance(p_class, mv_param):
                p_dict = {__p: {'values': {}, 'dist': True}}
                self.params.update(p_dict)
        for __c in __currently_defined_choices:
            c_class = globals()[__c]()
            if isinstance(c_class, choice_param):
                c_dict = {__c: {'values': {}, 'dist': False}}
                self.choices.update(c_dict)

        if template is not None:
            for tmp_type in os.listdir(template):
                with open(os.path.join(template, tmp_type), 'rb') as f:
                    tmp = pickle.load(f)
                    for param_name, param_info in tmp.items():
                        if param_name != 'num':
                            p_v = param_info['values'].copy()
                            for k, v in param_info['values'].items():
                                new_sub_name = '-'.join([tmp_type.split('.')[0].split('_')[-1], str(k)])
                                p_v[new_sub_name] = p_v.pop(k)
                            if param_name in self.params.keys():
                                self.params[param_name]['values'].update(p_v)
                            else:
                                param_info['values'] = p_v
                                self.params[param_name] = param_info
                        else:
                            if int(tmp_type.split('.')[0].split('_')[-1]) == 2:
                                self.params['num_axons'] = {'values': {'1-1': param_info['values'][1]}, 'dist': True}
                            if int(tmp_type.split('.')[0].split('_')[-1]) == 3:
                                self.params['num_apical_dendrites'] = {'values': {'1-1': param_info['values'][1]}, 'dist': True}
                            if int(tmp_type.split('.')[0].split('_')[-1]) == 4:
                                self.params['num_basal_dendrites'] = {'values': {'1-1': param_info['values'][1]}, 'dist': True}

    def update_param(self, param_name: str, value: tuple, type_id: int=2, order: int=1) -> None:
        key_name = '-'.join([str(type_id), str(order)])
        self.params[param_name]['values'][key_name] = value

    def get_params(self, param_name: str, type_id: int=1, order: int=1, target_id: int=-1) -> dict:
        if param_name not in self.params:
            try:
                result_param = globals()[param_name]
                return result_param()
            except KeyError as e:
                print('parameter {} not defined'.format(e))
        else:
            values = interp_param(all_params=self.params, param_name=param_name, type_id=type_id, order=order, target_id=target_id)
            result_param = globals()[param_name]
            updated_param = result_param(values=values)
            return updated_param

    def update_choices(self, choice_name: str, value: tuple, type_id: int=2, order: int=1) -> None:
        key_name = '-'.join([str(type_id), str(order)])
        self.choices[choice_name]['values'][key_name] = value

    def get_choices(self, param_name: str, type_id: int=1, order: int=1, target_id: int=-1) -> dict:
        if param_name not in self.choices:
            try:
                result_param = globals()[param_name]
                return result_param()
            except KeyError as e:
                print('parameter {} not defined'.format(e))
        else:
            values = interp_param(all_params=self.choices, param_name=param_name, type_id=type_id, order=order, target_id=target_id)
            result_param = globals()[param_name]
            updated_param = result_param(values=values)
            return updated_param

    def initialize_axons(self, **kwargs) -> None:
        """Initialize axons."""
        name_id = {'num_axons': 2, 'num_apical_dendrites': 3, 'num_basal_dendrites': 4}
        axon_count = 0
        self.axons = {_id: {} for _id in name_id.values()}

        for axon_name in name_id.keys():
            if self.params[axon_name]['values']:
                num = self.get_params(axon_name, type_id=1, order=1)
                for _ in range(num.sample(1)):
                    type_id = name_id[axon_name]
                    _axon_params = axon_params()
                    _axon_params.update_params(id=axon_count, type_id=type_id)
                    self.axons[type_id].update({axon_count: _axon_params})
                    axon_count += 1

    def gui_binding(self, binder) -> None:
        binder.collect()
        param_name = binder.name
        bind_dicts = globals()[param_name].collect_from_gui(binder)
        for bd in bind_dicts:
            if binder.__class__.__name__.endswith('mv'):
                self.params[param_name]['values'].update(bd)
            else:
                self.choices[param_name]['values'].update(bd)

    def load_temp(self, temp: str) -> None:
        self.__init__(temp)

    def refresh(self) -> None:
        self.__init__(self.template)


def uniform_dist(low: float=0.0, high: float=1.0, num: int=1, **kwargs) -> np.ndarray:
    if 'avg' in kwargs and 'size' in kwargs:
        window = kwargs['size'] * 2
        low = kwargs['avg'] - kwargs['size']
    else:
        window = high - low

    result = np.random.rand(num) * window + low
    if num == 1:
        return result[0]
    else:
        return result

def lognorm_dist(mu: float=0.0, sigma: float=0.0, num: int=1, loc: float=0.0, symm: bool=False) -> np.ndarray:
    log_norm_dist = stats.lognorm(s=sigma, scale=np.exp(mu), loc=loc)
    dist = log_norm_dist.rvs(size=num)
    if symm:
        signs = np.array([np.random.choice([1, -1]) for _ in range(num)])
        dist *= signs
    if num == 1:
        return dist[0]
    return dist

def get_dist(type: str='uniform', num: int=1, *args, **kwargs) -> Any:
    """Overall distribution function for easier switch"""
    mean, var = (1, 1)

    if not type.startswith('bi') and '-' not in type:
        if len(args) > 0:
            mean = args[0]
            var = args[1]
    elif 'lognorm' in type:
        if len(args) == 7:
            w = args[0]
            mean_1 = args[1]
            var_1 = args[2]
            loc_1 = args[3]
            mean_2 = args[4]
            var_2 = args[5]
            loc_2 = args[6]
        else:
            w = args[0]
            mean_1 = args[1]
            var_1 = args[2]
            mean_2 = args[3]
            var_2 = args[4]
            loc_1 = 0.0
            loc_2 = 0.0
    else:
        w = args[0]
        mean_1 = args[1]
        mean_2 = args[3]
        var_1 = args[2]
        var_2 = args[4]

    if type == 'uniform':
        if 'low' in kwargs:
            low = kwargs['low']
            high = kwargs['high']
            return uniform_dist(low=low, high=high, num=num)
        else:
            kws = {'avg': mean, 'size': var}
            return uniform_dist(num=num, **kws)
    elif type == 'norm':
        results = stats.norm(loc=mean, scale=var).rvs(size=num)
        if num == 1:
            results = results[0]
        return results

    elif type == 'lognorm':
        if len(args) > 2:
            loc = args[2]
        else:
            loc = 0
        if 'symm' in kwargs:
            return lognorm_dist(mu=mean, sigma=var, num=num, symm=kwargs['symm'], loc=loc)
        else:
            return lognorm_dist(mu=mean, sigma=var, num=num, loc=loc)

    elif type == 'bi-lognorm':
        temp_num = num * 100
        dist1_num = int(temp_num * w)
        dist2_num = temp_num - dist1_num
        dist_1 = lognorm_dist(mu=mean_1, sigma=var_1, loc=loc_1, num=dist1_num)
        dist_2 = lognorm_dist(mu=mean_2, sigma=var_2, loc=loc_2, num=dist2_num)
        if not isinstance(dist_2, np.ndarray):
            dist_2 = np.array([dist_2])
        if not isinstance(dist_1, np.ndarray):
            dist_1 = np.array([dist_1])
        mixed_dist = np.concatenate((dist_1, dist_2))
        np.random.shuffle(mixed_dist)
        return mixed_dist[:num]

    elif type == 'exponential-uniform':
        temp_num = num * 100
        dist1_num = int(temp_num * w)
        dist2_num = temp_num - dist1_num
        dist_1 = stats.expon(loc=mean_1, scale=var_1).rvs(dist1_num)
        dist_2 = stats.uniform(loc=mean_2, scale=var_2).rvs(dist2_num)
        mixed_dist = np.concatenate((dist_1, dist_2))
        np.random.shuffle(mixed_dist)
        return mixed_dist[:num]

    elif type == 'exponential':
        return stats.expon.rvs(loc=mean, scale=var, size=num)

    elif type == 'bi-vonmises':
        ratio = w
        vm_1_num = int(num * ratio)
        vm_2_num = num - int(num * ratio)
        vm_1 = stats.vonmises(loc=mean_1, kappa=var_1).rvs(vm_1_num)
        vm_2 = stats.vonmises(loc=mean_2, kappa=var_2).rvs(vm_2_num)
        vm_dist = np.concatenate((vm_1, vm_2))
        np.random.shuffle(vm_dist)
        return vm_dist

    elif type == 'vonmises':
        vm = stats.vonmises(loc=mean, kappa=var + 0.001).rvs(num)
        np.random.shuffle(vm)
        return vm

    elif type == 'uniform-vonmises':
        first_uni = get_dist(mean=mean[0], var=var[0], num=num, type='uniform')
        second_vm = get_dist(mean=mean[1], var=var[1], num=num, type='vonmises')
        return first_uni, second_vm

    elif type == 'vonmises_fisher':
        loc = vec_from_sph(*args[0])
        kappa = args[1]
        return stats.vonmises_fisher(mu=loc, kappa=kappa).rvs(num)

    else:
        raise ValueError('Distribution {} is not defined.'.format(type))


def fit_dist(data: np.ndarray, model: str='lognorm', exclude: float=0.05) -> Tuple:
    """
    Fit 1 dimensional data into different statistical distributions
    :param exclude: exclude extreme values
    :param data: datapoint
    :param model: name of statistical model
    :return: mean and variance
    """
    if np.squeeze(data).ndim == 1:
        data_sorted = np.sort(data)
        data = data_sorted[int(len(data_sorted) * exclude):int(len(data_sorted) * (1 - exclude))]
    if model == 'lognorm':
        shape, loc, scale = stats.lognorm.fit(data)
        return scale, shape, loc

    if model == 'bi-lognorm':

        def bimodal_lognormal(x: float, w1: float, mu1: float, sigma1: float, mu2: float, sigma2: float) -> np.ndarray:
            """Bimodal lognormal."""
            w2 = 1 - w1
            dist1 = w1 * stats.lognorm.pdf(x, sigma1, scale=np.exp(mu1))
            dist2 = w2 * stats.lognorm.pdf(x, sigma2, scale=np.exp(mu2))
            return dist1 + dist2

        initial_guess = [1, 1, 1, 1, 1]
        bin_heights, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        w1, mu1, sigma1, mu2, sigma2 = curve_fit(bimodal_lognormal, bin_centers, bin_heights, p0=initial_guess)
        return w1, mu1, sigma1, mu2, sigma2

    if model == 'exponential_uniform':

        def exponential_uniform(x: float, w1: float, loc1: float, scale1: float, loc2: float, scale2: float) -> Tuple:
            """Exponential uniform."""
            w2 = 1 - w1
            dist1 = w1 * stats.expon.pdf(x, loc=loc1, scale=scale1)
            dist2 = w2 * stats.uniform.pdf(x, loc=loc2, scale=scale2)
            return dist1 + dist2
        initial_guess = [1, 1, 1, 1, 1]
        bin_heights, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        w1, mu1, sigma1, mu2, sigma2 = curve_fit(exponential_uniform, bin_centers, bin_heights, p0=initial_guess)
        return w1, mu1, sigma1, mu2, sigma2

    elif model == 'exponential':
        loc, scale = stats.expon.fit(data)
        return loc, scale

    elif model == 'normal':
        mu, sigma = stats.norm.fit(data)
        return mu, sigma

    elif model == 'uniform':
        return float((np.min(data) + np.max(data)) / 2), float((np.max(data) - np.min(data)) / 2)

    elif model == 'vonmises_fisher':
        assert data.ndim == 2 and data.shape[1] == 3, 'wrong shape for 3d vector data'
        mu, kappa = stats.vonmises_fisher.fit(data)
        return mu, kappa

    elif model == 'vonmises':
        kappa, loc, scale = stats.vonmises.fit(data)
        return loc, kappa

    else:
        raise ValueError('Distribution {} is not defined.'.format(model))


def interp(y: List, x: List, pos: float) -> float:
    if len(x) > 1:
        x = np.concatenate(([0], np.array(x)))
        norm_value = np.cumsum(x) / np.sum(x)
        norm_value = norm_value[:-1]
        f = interp1d(x=norm_value, y=y, fill_value='extrapolate')
        return f(pos)
    else:
        return y[0]


def interp_param(all_params: dict, param_name: str, type_id: int=1, order: int=1, target_id: int=-1) -> Any:
    """
    Interpolate parameters if the direct neurite type and order are not given. Will interpolate to the nearst type
    and order. Example: type 2 order 5 -> (2, 5) will be interpolated to (3, 4) given [(3, 4), (4, 5), (3, 8)].
    """

    avail_param = all_params[param_name]['values']
    if target_id >= 0:
        order = target_id

    avail_types = np.unique([int(s.split('-')[0]) for s in avail_param])
    if type_id not in avail_types:
        type_id = avail_types[np.argmin(np.abs(avail_types - type_id))]
    type_params = [s for s in avail_param if s.startswith('{}'.format(type_id))]

    avail_order = np.unique([int(s.split('-')[1]) for s in type_params])
    if order not in avail_order:
        order = avail_order[np.argmin(np.abs(avail_order - order))]

    return avail_param['-'.join([str(type_id), str(order)])]

