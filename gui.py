from typing import Any, List, Tuple
from components.neuron import Neuron
from utils.params_var import *
from utils.utils import write_swc
import pickle
from nicegui import ui, binding
import os
import numpy as np
import math
from cloudvolume import Skeleton
from datetime import datetime

__TEMP_DIR = 'swc'


class _basic_mv_binder:

    def __init__(self, name: str) -> None:
        """
        Basic binding handle for all mean-variance based parameters
        """
        self.name = name
        self.mean = {}
        self.var = {}
        self.type = {}
        self.order = {}
        self.dist_type = globals()[name].dist_type
        self.count = 0

    def collect(self) -> None:
        """
        Gather different sets of the same parameter
        """
        for i in range(1, self.count + 1):
            self.type.update({i: self.__getattribute__('type_' + str(i))})
            self.order.update({i: self.__getattribute__('order_' + str(i))})

    def __str__(self) -> str:
        return 'It has types' + '-'.join([str(j) for j in list(self.type.keys())])


class _basic_choice_binder:

    def __init__(self, name: str) -> None:
        """
        Basic binding handle for all mean-variance based parameters
        """
        self.name = name
        self.choice = {}
        self.type = {}
        self.order = {}
        self.count = 0

    def info_from_mv(self, mv_binder: Any) -> None:
        self.type.update(mv_binder.type)
        self.order.update(mv_binder.order)

    def collect(self) -> None:
        """
        Gather different sets of the same parameter
        """
        for i in range(1, self.count + 1):
            self.choice.update({i: self.__getattribute__('choice_' + str(i))})
            self.type.update({i: self.__getattribute__('type_' + str(i))})
            self.order.update({i: self.__getattribute__('order_' + str(i))})

    def __str__(self) -> str:
        return 'It has ' + '-'.join([str(j) for j in list(self.type.keys())])


class _single_mv(_basic_mv_binder):
    """
    Includes lognorm, norm, uniform, exponential distribution, that contain one mean and one variance
    """
    def __init__(self, name: str) -> None:
        """Initialize the object."""
        super().__init__(name)

    def collect(self) -> None:
        super().collect()
        for i in range(1, self.count + 1):
            self.mean.update({i: self.__getattribute__('mean_' + str(i))})
            self.var.update({i: self.__getattribute__('var_' + str(i))})


class _single_no_type_mv:

    def __init__(self, name: str) -> None:
        self.name = name
        self.mean = {}
        self.var = {}
        self.dist_type = globals()[name].dist_type

    def collect(self) -> None:
        self.mean.update({0: self.__getattribute__('mean_')})
        self.var.update({0: self.__getattribute__('var_')})


class _bi_mv(_basic_mv_binder):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.ratio = {}

    def collect(self) -> None:
        super().collect()
        for i in range(1, self.count + 1):
            self.ratio.update({i: self.__getattribute__('ratio_' + str(i))})
            self.mean.update({i: [self.__getattribute__('first_mean_' + str(i)), self.__getattribute__('second_mean_' + str(i))]})
            self.var.update({i: [self.__getattribute__('first_var_' + str(i)), self.__getattribute__('second_var_' + str(i))]})


class _vonmises_mv(_basic_mv_binder):
    """
    Includes lognorm, norm, uniform, exponential distribution, that contain one mean and one variance
    """
    def __init__(self, name: str, dim: int=2) -> None:
        """Initialize the object."""
        super().__init__(name)
        self.dim = dim
        assert dim in [2, 3], '2D or 3D'

    def collect(self) -> None:
        super().collect()
        for i in range(1, self.count + 1):
            if self.dim == 3:
                self.mean.update({i: (self.__getattribute__('x_' + str(i)), self.__getattribute__('y_' + str(i)), self.__getattribute__('z_' + str(i)))})
            elif self.dim == 2:
                self.mean.update({i: self.__getattribute__('theta_' + str(i))})
            self.var.update({i: self.__getattribute__('kappa_' + str(i))})


class _single_choice(_basic_choice_binder):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def collect(self) -> None:
        super().collect()


def binder_wrapper(binder_instance: Any, choice_instance: _basic_choice_binder=None) -> Any:
    """
    defined 4 types of callbacks but can be expanded later
    """

    def add_label_and_input_single() -> None:
        """
        for single medium and mean
        """
        binder_instance.count += 1
        if choice_instance:
            choice_instance.count += 1
        with ui.row().classes('w-full'):
            if binder_instance.dist_type.endswith('lognorm'):
                ui.input(label='Enter {} mean'.format(binder_instance.name), value='{:.3f}'.format(np.exp(globals()[binder_instance.name].values[0]))).bind_value(binder_instance, 'mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            else:
                ui.input(label='Enter {} mean'.format(binder_instance.name), value='{:.3f}'.format(globals()[binder_instance.name].values[0])).bind_value(binder_instance, 'mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Enter {} var'.format(binder_instance.name), value='{:.3f}'.format(globals()[binder_instance.name].values[1])).bind_value(binder_instance, 'var_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            if binder_instance.name == 'max_order':
                ui.select(options=['c', 's'], value='c', label='Pick an order scheme').classes('w-40').bind_value(choice_instance, 'choice_{}'.format(choice_instance.count)).style('margin-top: -20px;')
                ui.input(label='Specify type', value='2').bind_value(binder_instance, 'type_{}'.format(binder_instance.count)).bind_value(choice_instance, 'type_{}'.format(choice_instance.count)).style('margin-top: -20px;')
                ui.input(label='Specify order', value='1').bind_value(binder_instance, 'order_{}'.format(binder_instance.count)).bind_value(choice_instance, 'order_{}'.format(choice_instance.count)).style('margin-top: -20px;')
            else:
                ui.input(label='Specify type', value='2').bind_value(binder_instance, 'type_{}'.format(binder_instance.count)).style('margin-top: -20px;')
                ui.input(label='Specify order', value='1').bind_value(binder_instance, 'order_{}'.format(binder_instance.count)).style('margin-top: -20px;')

    def add_label_and_input_vmf() -> None:
        """
        for circular type of distribution
        """
        binder_instance.count += 1
        random_init_vec = np.random.rand(3) * 2 - 1
        with ui.row().classes('w-full'):
            if binder_instance.dim == 3:
                ui.input(label='Enter x', value='{:.3f}'.format(random_init_vec[0])).bind_value(binder_instance, 'x_{}'.format(binder_instance.count)).style('margin-top: -20px;')
                ui.input(label='Enter y', value='{:.3f}'.format(random_init_vec[1])).bind_value(binder_instance, 'y_{}'.format(binder_instance.count)).style('margin-top: -20px;')
                ui.input(label='Enter z', value='{:.3f}'.format(random_init_vec[2])).bind_value(binder_instance, 'z_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            else:
                ui.input(label='Enter theta', value='{:.3f}'.format(random_init_vec[0])).bind_value(binder_instance, 'theta_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Enter concentration', value=str(globals()[binder_instance.name].values[1])).bind_value(binder_instance, 'kappa_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Specify type', value='2').bind_value(binder_instance, 'type_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Specify order', value='1').bind_value(binder_instance, 'order_{}'.format(binder_instance.count)).style('margin-top: -20px;')

    def add_label_and_input_bi() -> None:
        """
        for bimodal distribution
        """
        binder_instance.count += 1
        with ui.row().classes('w-full'):
            ui.input(label='Enter ratio', value='{:.3f}'.format(globals()[binder_instance.name].values[0])).bind_value(binder_instance, 'ratio_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            if binder_instance.dist_type.endswith('lognorm'):
                ui.input(label='Enter 1st mean', value='{:.3f}'.format(np.exp(globals()[binder_instance.name].values[1]))).bind_value(binder_instance, 'first_mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            elif binder_instance.dist_type.endswith('vonmises'):
                ui.input(label='Enter 1st theta', value='{:.3f}'.format(globals()[binder_instance.name].values[1])).bind_value(binder_instance, 'first_mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            else:
                ui.input(label='Enter 1st mean', value='{:.3f}'.format(globals()[binder_instance.name].values[1])).bind_value(binder_instance, 'first_mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Enter 1st var', value='{:.3f}'.format(globals()[binder_instance.name].values[2])).bind_value(binder_instance, 'first_var_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            if binder_instance.dist_type.endswith('lognorm'):
                ui.input(label='Enter 2nd mean', value='{:.3f}'.format(np.exp(globals()[binder_instance.name].values[3]))).bind_value(binder_instance, 'second_mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            else:
                ui.input(label='Enter 2nd mean', value='{:.3f}'.format(globals()[binder_instance.name].values[3])).bind_value(binder_instance, 'second_mean_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Enter 2nd var', value=str(globals()[binder_instance.name].values[4])).bind_value(binder_instance, 'second_var_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Specify type', value='2').bind_value(binder_instance, 'type_{}'.format(binder_instance.count)).style('margin-top: -20px;')
            ui.input(label='Specify order', value='1').bind_value(binder_instance, 'order_{}'.format(binder_instance.count)).style('margin-top: -20px;')

    def add_label_and_input_no_type() -> None:
        """Add the label and input no type."""
        ui.label('{}'.format(binder_instance.name)).style('font-size: 16px; font-weight: bold; color: #518496;').style('margin-top: 20px;')
        with ui.column():
            ui.input(label='Enter {} mean'.format(binder_instance.name), value='{:.3f}'.format(globals()[binder_instance.name].values[0])).bind_value(binder_instance, 'mean_')
            ui.input(label='Enter {} var'.format(binder_instance.name), value='{:.3f}'.format(globals()[binder_instance.name].values[1])).bind_value(binder_instance, 'var_')
    if binder_instance.name in ['num_axons', 'num_apical_dendrites', 'num_basal_dendrites']:
        return add_label_and_input_no_type
    if binder_instance.dist_type in ['norm', 'lognorm', 'uniform']:
        return add_label_and_input_single
    elif binder_instance.dist_type in ['bi-lognorm', 'bi-vonmises', 'exponential-uniform']:
        return add_label_and_input_bi
    elif binder_instance.dist_type in ['vonmises', 'vonmises_fisher']:
        return add_label_and_input_vmf

def basic_pm_block(defined_mv_binder: Any, defined_choice_binder: Any=None) -> None:
    """Basic pm block."""
    with ui.expansion(defined_mv_binder.name, icon='expand_more', value=True).style('font-size: 20px; font-weight: 500; color:#329c94; margin-top: -20px;').classes('w-full'):
        if defined_choice_binder:
            with ui.card().style('background-color: #f5f5f5;'):
                binder_wrapper(defined_mv_binder, defined_choice_binder)()
                ui.button('Add another parameter', on_click=binder_wrapper(defined_mv_binder, defined_choice_binder)).style('font-size: 12px; color: black, background-color: #daf5ed;')
        else:
            with ui.card().style('background-color: #f5f5f5;'):
                binder_wrapper(defined_mv_binder)()
                ui.button('Add another parameter', on_click=binder_wrapper(defined_mv_binder)).style('font-size: 12px; color: black, background-color: #daf5ed;')


@binding.bindable_dataclass
class RandomSeedOption:
    value: int


def get_random_seed() -> int:
    """Return the random seed."""
    return int(np.random.rand() * 1000000000)


def check_templates(root_dir: str='swc') -> List:
    """Check the templates."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError('Directory {} does not exist.'.format(root_dir))

    temps = []
    all_avail = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and 'generated' not in f]
    for f in all_avail:
        p_f = os.path.join(root_dir, f, 'params')
        if len(os.listdir(p_f)) > 0:
            temps.append(f)
    return temps


def set_camera_location(scene: ui.scene, swc: Skeleton, scaling_factor: float) -> None:
    """Set the camera location."""
    max_x = -math.inf
    max_y = -math.inf
    max_z = -math.inf
    min_x = math.inf
    min_y = math.inf
    min_z = math.inf
    for v in swc.vertices:
        max_x = max(max_x, v[0])
        min_x = min(min_x, v[0])
        max_y = max(max_y, v[1])
        min_y = min(min_y, v[1])
        max_z = max(max_z, v[2])
        min_z = min(min_z, v[2])
    max_x /= scaling_factor
    min_x /= scaling_factor
    max_y /= scaling_factor
    min_y /= scaling_factor
    max_z /= scaling_factor
    min_z /= scaling_factor
    cam_x = (max_x - min_x) / 8
    cam_y = (max_y - min_y) / 8
    cam_z = (max_z - min_z) / 8
    scene.move_camera(max_x, max_y, max_z, -cam_x, -cam_y, -cam_z)


async def render(scene: ui.scene, swc: Skeleton) -> Any:
    scaling_factor = 20

    def scale(v: float) -> float:
        """Scale."""
        return v / scaling_factor

    for i, v in enumerate(swc.vertices):
        if swc.vertex_types[i] == 1:
            color = '#00FF00'
        elif swc.vertex_types[i] == 2:
            color = '#FF0000'
        elif swc.vertex_types[i] == 3:
            color = '#0000FF'
        elif swc.vertex_types[i] == 4:
            color = '#00FFFF'
        else:
            color = '#a88332'
        scene.sphere(radius=scale(swc.radius[i])).move(scale(v[0]), scale(v[1]), scale(v[2])).material(color)

    for e in swc.edges:
        start = scale(swc.vertices[e[0]])
        end = scale(swc.vertices[e[1]])
        scene.line(start, end).material('#000000')

    set_camera_location(scene, swc, scaling_factor)

def main() -> None:

    nrn = Neuron()
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    session_info = {'name': time_now}

    with ui.grid(columns=2).classes('w-full'):
        scene = ui.scene(grid=False).classes('w-full h-svh')
        nmtp = neuron_params()
        _num_axons_flat = _single_no_type_mv('num_axons')
        _num_apical_dendrites = _single_no_type_mv('num_apical_dendrites')
        _num_basal_dendrites = _single_no_type_mv('num_basal_dendrites')
        _max_order_flat = _single_mv('max_order')
        _num_segments_flat = _single_mv('num_segments')
        _symmetry_flat = _single_mv('symmetry')
        _d2h_ratio_flat = _single_mv('d2h_ratio')
        _init_vec_flat = _vonmises_mv('init_vec', dim=3)
        _init_r_flat = _single_mv('init_r')
        _contr_factor_flat = _single_mv('contr_factor')
        _length_flat = _bi_mv('length')
        _hist_w_flat = _single_mv('hist_w')
        _proj_w_flat = _single_mv('proj_w')
        _phi_flat = _single_mv('phi')
        _theta_flat = _bi_mv('theta')
        _step_size_flat = _single_mv('step_size')
        _ampl_flat = _single_mv('ampl')
        _tilt_flat = _single_mv('tilt')
        _ratio_flat = _bi_mv('ratio')
        _roll_flat = _vonmises_mv('roll', dim=2)
        _order_scheme = _single_choice('order_scheme')
        with ui.column().classes('w-full'):
            with ui.scroll_area().classes('h-[calc(100vh-100px)]'):
                with ui.row():
                    ui.input(label='Enter session name', value=time_now).bind_value(session_info, 'name')
                    ui.button('CREATE', on_click=lambda: nrn.from_scratch(session_info['name']), color='#569599').classes('px-1 py-0 text-lg')
                with ui.expansion('Neuron related parameters', icon='expand_more', value=True).style('font-size: 28px; font-weight: 500; color: #062e2a'):
                    with ui.row():
                        binder_wrapper(_num_axons_flat)()
                        binder_wrapper(_num_apical_dendrites)()
                        binder_wrapper(_num_basal_dendrites)()
                ui.separator()
                with ui.expansion('Single axon/dendrite related parameters', icon='expand_more', value=True).style('font-size: 28px; font-weight: 500; color: #062e2a'):
                    basic_pm_block(_max_order_flat, _order_scheme)
                    basic_pm_block(_num_segments_flat)
                    basic_pm_block(_symmetry_flat)
                    basic_pm_block(_d2h_ratio_flat)
                    basic_pm_block(_init_vec_flat)
                    basic_pm_block(_init_r_flat)
                ui.separator()
                with ui.expansion('Branch segment related parameters', icon='expand_more', value=True).style('font-size: 28px; font-weight: 500; color: #062e2a'):
                    basic_pm_block(_length_flat)
                    basic_pm_block(_hist_w_flat)
                    basic_pm_block(_proj_w_flat)
                    basic_pm_block(_contr_factor_flat)
                ui.separator()
                with ui.expansion('Node related parameters', icon='expand_more', value=True).style('font-size: 28px; font-weight: 500; color: #062e2a'):
                    basic_pm_block(_phi_flat)
                    basic_pm_block(_theta_flat)
                    basic_pm_block(_step_size_flat)
                ui.separator()
                with ui.expansion('Bifurcation related parameters', icon='expand_more', value=True).style('font-size: 28px; font-weight: 500; color: #062e2a'):
                    basic_pm_block(_ampl_flat)
                    basic_pm_block(_tilt_flat)
                    basic_pm_block(_ratio_flat)
                    basic_pm_block(_roll_flat)

            async def generate() -> None:
                """Generate."""
                nmtp.refresh()
                np.random.seed(random_seed.value)
                nmtp.gui_binding(_num_axons_flat)
                nmtp.gui_binding(_num_apical_dendrites)
                nmtp.gui_binding(_num_basal_dendrites)
                nmtp.gui_binding(_max_order_flat)
                nmtp.gui_binding(_num_segments_flat)
                nmtp.gui_binding(_symmetry_flat)
                nmtp.gui_binding(_d2h_ratio_flat)
                nmtp.gui_binding(_length_flat)
                nmtp.gui_binding(_init_vec_flat)
                nmtp.gui_binding(_init_r_flat)
                nmtp.gui_binding(_hist_w_flat)
                nmtp.gui_binding(_proj_w_flat)
                nmtp.gui_binding(_contr_factor_flat)
                nmtp.gui_binding(_phi_flat)
                nmtp.gui_binding(_theta_flat)
                nmtp.gui_binding(_step_size_flat)
                nmtp.gui_binding(_ampl_flat)
                nmtp.gui_binding(_tilt_flat)
                nmtp.gui_binding(_roll_flat)
                nmtp.gui_binding(_ratio_flat)
                nmtp.gui_binding(_order_scheme)
                nrn.create(updated_params=nmtp, create_exact=False)
                nrn.grow()
                nrn.assemble()
                swc = nrn.skl
                scene.clear()
                await render(scene, swc)
                random_seed.value = get_random_seed()
                print('neuron generated!')

            async def save() -> None:
                """Save."""
                with open(os.path.join('swc', session_info['name'], 'params', 'params.pkl'), 'wb') as f:
                    pickle.dump(nmtp, f)
                nrn.write_swc(os.path.join('swc', session_info['name'], 'generated', '0.swc'), flip_xz=False)
                print('saved!')

            async def mass_generate() -> None:
                """Mass generate."""
                for i in range(10):
                    nrn.create()
                    nrn.grow()
                    with open(os.path.join('swc', session_info['name'], 'generated', '{}.pkl'.format(i)), 'wb') as f:
                        pickle.dump(nrn, f)
                    write_swc(os.path.join('swc', session_info['name'], 'generated', '{}.swc'.format(i)), nrn.skl)
            random_seed = RandomSeedOption(value=get_random_seed())
            with ui.row().classes('w-full justify-end'):
                ui.button('GENERATE', on_click=generate, color='#569599').classes('px-2 py-1 text-lg')
                ui.button('SAVE', on_click=save, color='#569599').classes('px-2 py-1 text-lg')
                ui.button('MASS GENERATE', on_click=mass_generate, color='#569599').classes('px-2 py-1 text-lg')

        ui.run(title='Neuron Generator')


if __name__ in {'__main__', '__mp_main__'}:
    main()
