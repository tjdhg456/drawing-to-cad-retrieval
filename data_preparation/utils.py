import numpy as np
import matplotlib.pyplot as plt
import numpy
from numpy.linalg import norm
from mpl_toolkits.mplot3d import axes3d
import trimesh
import trimesh.transformations as transform
import os
import math
from itertools import combinations

def plot_vox(view_angle, mats, title='base', out_path=None):
    num_mats = len(mats)
    fig = plt.figure(figsize=(4*num_mats, 4))
    for i, mat in enumerate(mats):
        # ax = Axes3D(fig)
        ax = fig.add_subplot(1, num_mats, i+1, projection='3d')
        ax.view_init(view_angle[0], view_angle[1])
        ax.voxels(mat, edgecolor='k')
        plt.title(title + ' ' + str(i))
        if out_path is not None:
            plt.savefig(out_path)
            plt.close('all')
            print('%s is saved' %out_path)
        else:
            plt.show()

def load_mesh(obj_path):
    mesh = trimesh.load_mesh(obj_path, file_type='obj', process=False)

    def as_mesh(scene_or_mesh):
        """
        Convert a possible scene to a mesh.

        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                          for g in scene_or_mesh.geometry.values()))
        else:
            assert (isinstance(scene_or_mesh, trimesh.Trimesh))
            mesh = scene_or_mesh
        return mesh

    mesh = as_mesh(mesh)
    return mesh

## Rotate the obj files and save
def rotate(obj_path, out_path, angle_list, base=False, random=False):
    mesh = load_mesh(obj_path)
    mesh.vertices -= mesh.center_mass

    # Calculate the momentum and principal axes
    inertia = mesh.moment_inertia

    if base == True:
        p_axis = Principal_Axes(inertia)
        p_R = np.concatenate((np.concatenate(p_axis, axis=0).reshape(-1, 3), np.array([[0., 0., 0.]])), axis=0)
        R = np.concatenate((p_R, np.array([[0.], [0.], [0.], [1]])), axis=1)

    else:
        if random == False:
            alpha, beta, gamma = np.radians(angle_list[0]), np.radians(angle_list[1]),\
                                 np.radians(angle_list[2])

            Rx = transform.rotation_matrix(alpha, [1, 0, 0])
            Ry = transform.rotation_matrix(beta, [0, 1, 0])
            Rz = transform.rotation_matrix(gamma, [0, 0, 1])
            R = transform.concatenate_matrices(Rx, Ry, Rz)

        else:
            # Random Rotation Matrix
            R = transform.random_rotation_matrix()

    # Rotation
    mesh.apply_transform(R)
    mesh.export(out_path)

## For Calculating principal axes
def Principal_Axes(inertia):
    e_values, e_vectors = numpy.linalg.eig(inertia)

    # axis1 is the principal axis with the biggest eigen value (eval1)
    # axis2 is the principal axis with the second biggest eigen value (eval2)
    # axis3 is the principal axis with the smallest eigen value (eval3)
    order = numpy.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()
    return axis1, axis2, axis3

## For Group Genaration
def Extract_With_Combination(adjacency_matrix=np.ndarray):
    '''
    For each level(1~num_parts), Generate all possible combinations (order-independent)
       Except level 1 or num_parts; add all to resulting list
    For each level except for 1 or num_parts, Let L: current level.
    Generate combination(L, 2)
    Check if all pairs are connected (using adjacency matrix)
    '''

    # list_part_indices = {i for i in range(len(dict_mesh_subgroup))} # set
    # print(list_part_indices)

    MAX_DEPTH = adjacency_matrix.shape[0]
    list_part_indices_iota = [i for i in range(MAX_DEPTH)]

    print('MAX Depth : %d' % MAX_DEPTH)
    result = [[]] * MAX_DEPTH
    for level in range(MAX_DEPTH):
        combinations_all_at_level = combinations(list_part_indices_iota, level + 1)
        result[level] = __CalculateConnectedCombinations(list(combinations_all_at_level), level + 1, adjacency_matrix)

    return result

def __CalculateConnectedCombinations(list_combinations, num_elements_in_each_combination, adjacency_matrix):
    result_list_connected_combinations = []

    # print('[DEBUG] list_combinations at level', num_elements_in_each_combination, ' = ', list_combinations)

    for target_combination in list_combinations:
        # Indices of to be connection-confirmed
        dict_connections_to_be_confirmed = {}
        for i in target_combination:
            dict_connections_to_be_confirmed[i] = False

        # print('[DEBUG] dict_connections_to_be_confirmed = ', dict_connections_to_be_confirmed)
        # exit()

        # Check adjacency for all combinations FIXME there're duplicated comparisons
        combination_pairs = combinations(target_combination, 2)
        for combination_pair in list(combination_pairs):
            if adjacency_matrix[combination_pair[0]][combination_pair[1]] == True or \
                    adjacency_matrix[combination_pair[1]][combination_pair[0]] == True:
                dict_connections_to_be_confirmed[combination_pair[0]] = True
                dict_connections_to_be_confirmed[combination_pair[1]] = True

        # Add to result if all elements in @var:dict_connections_to_be_confirmed is True
        if all(connected == True for connected in dict_connections_to_be_confirmed.values()):
            result_list_connected_combinations.append(target_combination)

    return result_list_connected_combinations