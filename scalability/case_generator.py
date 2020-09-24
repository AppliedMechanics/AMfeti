from amfe.mesh_creator import *
import matplotlib.pyplot as plt
import scipy
import pickle
from amfe.component import StructuralComponent, TreeBuilder
from amfe.material import KirchhoffMaterial
from amfe.forces import constant_force
from amfe.solver.translators import MulticomponentMechanicalSystem
import numpy as np


def create_subdomain_mesh_components(n_domains_x, n_domains_y, n_ele_x, n_ele_y, length, height, material, force,
                                     force_direction):
    subdomain_length = length / n_domains_x
    subdomain_height = height / n_domains_y
    x_index = 0
    y_index = 0
    mesh_dict = dict()
    components_list = []
    components_ids_list = []
    counter = 0
    for iy in np.arange(n_domains_y):
        x_index = 0
        for ix in np.arange(n_domains_x):
            msh_creator = RectangleMesh2D(subdomain_length, subdomain_height, n_ele_x, n_ele_y, x_index, y_index)
            msh_creator.generate_mesh()
            msh_creator.set_homogeneous_group('material')
            if x_index == 0:
                msh_creator.add_boundary('left', 'dirichlet')
            if x_index + subdomain_length == length:
                msh_creator.add_boundary('right', 'neumann')
            subdomain_mesh = msh_creator.mesh
            mesh_dict.update({counter: subdomain_mesh})

            counter = counter + 1

            component_subdomain = StructuralComponent(subdomain_mesh)
            ui.assign_material_by_group(component_subdomain, material, 'material')
            if x_index == 0:
                ui.set_dirichlet_by_group(component_subdomain, 'dirichlet', ('ux'), 'Dirichlet_x')
                ui.set_dirichlet_by_group(component_subdomain, 'dirichlet', ('uy'), 'Dirichlet_y')
            if x_index + subdomain_length == length:
                ui.set_neumann_by_group(component_subdomain, 'neumann', force_direction, False, 'Load', force)
            components_list.append(component_subdomain)
            components_ids_list.append(counter)
            x_index = x_index + subdomain_length
        y_index = y_index + subdomain_height
    return mesh_dict, components_ids_list, components_list


def _create_K_B_f_dict_splitsystem(B_dict, msys_dict):
    K_dict_trans = dict()
    K_effective_trans = dict()
    M_dict_trans = dict()
    B_dict_trans = dict()
    f_dict_trans = dict()
    f_effective_trans = dict()
    interface_dict = dict()
    B_dict_split = dict()
    int_num = 1

    for i_system, msys in msys_dict.items():
        subs_key = i_system
        K_dict_trans[subs_key] = msys.K(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        M_dict_trans[subs_key] = msys.M(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        f_dict_trans[subs_key] = msys.f_ext(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        K_effective_trans[subs_key] = scipy.sparse.vstack([
            scipy.sparse.hstack([K_dict_trans[subs_key] - omg2 * M_dict_trans[subs_key],
                                 omg * (aa * K_dict_trans[subs_key] + bb * M_dict_trans[subs_key])]),
            scipy.sparse.hstack([K_dict_trans[subs_key] - omg2 * M_dict_trans[subs_key],
                                 omg * (aa * K_dict_trans[subs_key] + bb * M_dict_trans[subs_key])])])
        f_effective_trans[subs_key] = np.append(np.zeros(f_dict_trans[subs_key].size), f_dict_trans[subs_key])

    for key, items in B_dict.items():
        items_coo = items.tocoo()
        items_coo.data = np.append(items_coo.data, items_coo.data)
        row_increment_factor = items_coo.row.max() + 1
        items_coo.row = np.append(items_coo.row, items_coo.row + row_increment_factor)
        items_coo.col = np.append(items_coo.col, items_coo.col + K_dict_trans[key[0]].get_shape()[1])
        B_dict_split[key] = scipy.sparse.coo_matrix((items_coo.data, (items_coo.row, items_coo.col)),
                                                    shape=(items_coo.shape[0] * 2, items_coo.shape[1] * 2))

    for i_system, msys in msys_dict.items():
        subs_key = i_system
        B_local = dict()
        for key in B_dict_split.keys():
            if key not in interface_dict:
                interface_dict[key] = 'interface' + str(int_num)
                interface_dict[(key[1], key[0])] = 'interface' + str(int_num)
                int_num += 1
            if key[0] == i_system:
                B_local[interface_dict[key]] = B_dict_split[key]
        B_dict_trans[subs_key] = B_local
    return K_dict_trans, M_dict_trans, B_dict_trans, f_dict_trans, K_effective_trans, f_effective_trans


def _create_K_B_f_dict(B_dict, msys_dict):
    K_dict_trans = dict()
    M_dict_trans = dict()
    B_dict_trans = dict()
    f_dict_trans = dict()
    interface_dict = dict()
    int_num = 1

    for i_system, msys in msys_dict.items():
        subs_key = i_system
        K_dict_trans[subs_key] = msys.K(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        M_dict_trans[subs_key] = msys.M(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        f_dict_trans[subs_key] = msys.f_ext(np.zeros(msys.dimension), np.zeros(msys.dimension), 0)
        B_local = dict()
        for key in B_dict.keys():
            if key not in interface_dict:
                interface_dict[key] = 'interface' + str(int_num)
                interface_dict[(key[1], key[0])] = 'interface' + str(int_num)
                int_num += 1
            if key[0] == i_system:
                B_local[interface_dict[key]] = B_dict[key]
        B_dict_trans[subs_key] = B_local
    return K_dict_trans, M_dict_trans, B_dict_trans, f_dict_trans


def create_subdomain_matrices(n_domains_x=3, n_domains_y=3, n_ele_x=10, n_ele_y=10, length=8, height=8,
                              split_system=False):
    E_alu = 210e9
    nu_alu = 0.3
    rho_alu = 7850
    material = KirchhoffMaterial(E_alu, nu_alu, rho_alu, thickness=1.0)

    force = constant_force(100000)
    force_direction = np.array([0.0, 1.0])
    mesh_dict, components_ids_list, components_list = create_subdomain_mesh_components(n_domains_x, n_domains_y,
                                                                                       n_ele_x, n_ele_y, length, height,
                                                                                       material, force, force_direction)
    tree_builder_components = TreeBuilder()
    tree_builder_components.add(components_ids_list, components_list)
    structural_composite_components = tree_builder_components.root_composite
    structural_composite_components.update_component_connections()
    substructured_system_components = MulticomponentMechanicalSystem(structural_composite_components, 'boolean',
                                                                     all_linear=True)
    substructured_system = substructured_system_components
    structural_composite = structural_composite_components

    if split_system:
        K_dict, M_dict, B_dict, f_dict, K_effective_dict, f_effective_dict = _create_K_B_f_dict_splitsystem(
            substructured_system.connections, substructured_system.mechanical_systems)
    else:
        K_dict, M_dict, B_dict, f_dict = _create_K_B_f_dict(substructured_system.connections,
                                                            substructured_system.mechanical_systems)
