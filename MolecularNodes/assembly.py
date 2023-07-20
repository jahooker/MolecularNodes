import bpy
import numpy as np
from . import nodes
import biotite.structure.io.pdbx as pdbx
import re
from typing import Iterator

def get_transformations_pdbx(file_pdbx) -> dict:
    # The output has an entry for each transformation,
    # indexable by the string integer of the assembly number (e.g. transform_dict['1'])
    # which contains tuple of the 3x3 rotation matrix and the 1x3 transform matrix
    return pdbx.convert._get_transformations(file_pdbx.get_category('pdbx_struct_oper_list'))

def get_transformations_pdb(file_pdb):
    # TODO Ideally, this would be handled by a dedicated pdb parser.
    # Lines that contain 'SMTRY' specify symmetry operations
    lines = np.array(file_pdb.lines)
    sym_lines = lines[np.char.rfind(lines, 'SMTRY') >= 0]

    float_pattern = re.compile('\d+\.\d+')
    sym_array = np.array([float_pattern.findall(line) for line in sym_lines],
                         dtype=np.float32)

    k = len(sym_lines)
    assert sym_array.shape == (k, 4) and k % 3 == 0, \
        "Symmetry operations shall be represented as 3x4 matrices!"
    for i in range(0, k, 3):
        yield (sym_array[i:i+3,:3],
               sym_array[i:i+3,3:].reshape((1, 3)))

def get_transformations_mmtf(assemblies) -> Iterator[np.ndarray[float]]:
    """
    Generates the transformations required to build out each biological assembly
    as a N×3×4 `numpy.ndarray` where N is the number of transformations.
    """
    for assembly in assemblies:
        transforms = assembly['transformList']
        n = len(transforms)
        # FIXME nx3x4 or nx4x4?
        yield np.array([transform['matrix'] for transform in transforms]).reshape((n, 4, 4))

def create_assembly_node(name, transformation_matrices: list[np.ndarray[float]]):

    if (node_mat := bpy.data.node_groups.get(f'MOL_RotTransMat_{name}')):
        return node_mat

    node_mat = nodes.gn_new_group_empty(f'MOL_RotTransMat_{name}')
    node_mat.inputs.remove(node_mat.inputs['Geometry'])
    node_mat.nodes['Group Output'].location = [800, 0]
    node_mat.outputs['Geometry'].name = 'RotTransMat'

    node_transforms = [nodes.rotation_matrix(
        node_group=node_mat, mat=mat, location=[0, 300 * -i]
    ) for i, mat in enumerate(transformation_matrices)]

    node_join = node_mat.nodes.new('GeometryNodeJoinGeometry')
    node_join.location = [300, 0]

    for node_transform in reversed(node_transforms):
        node_mat.links.new(node_transform.outputs['Geometry'], node_join.inputs['Geometry'])

    node_mat.links.new(node_join.outputs['Geometry'], node_mat.nodes['Group Output'].inputs['RotTransMat'])

    return node_mat

def create_biological_assembly_node(name, transformation_matrices: list):

    if (node_bio := bpy.data.node_groups.get(f'MOL_assembly_{name}')) is not None:
        return node_bio

    # try to create the assembly transformation nodes first,
    # so if they fail, nothing else is created
    data_trans = create_assembly_node(name, transformation_matrices)

    node_bio = nodes.gn_new_group_empty(f'MOL_assembly_{name}')

    node_input = node_bio.nodes[bpy.app.translations.pgettext_data("Group Input",)]
    node_output = node_bio.nodes[bpy.app.translations.pgettext_data("Group Output",)]
    node_output.location = [400, 0]

    node_assembly = nodes.add_custom_node_group_to_node(node_bio, 'MOL_utils_bio_assembly', location=(0, 0))

    node_trans = nodes.add_custom_node_group_to_node(node_bio, data_trans.name, location=(-400, -200))

    node_bio.links.new(node_input.outputs['Geometry'], node_assembly.inputs['Geometry'])
    node_bio.links.new(node_trans.outputs['RotTransMat'], node_assembly.inputs['RotTransMat'])
    node_bio.links.new(node_assembly.outputs['Instances'], node_output.inputs[0])

    node_bio.inputs.new('NodeSocketFloat', 'Scale Rotation')
    node_bio.inputs.get('Scale Rotation').default_value = 1
    node_bio.links.new(node_input.outputs['Scale Rotation'], node_assembly.inputs['Scale Rotation'])

    node_bio.inputs.new('NodeSocketFloat', 'Scale Translation')
    node_bio.inputs.get('Scale Translation').default_value = 1
    node_bio.links.new(node_input.outputs['Scale Translation'], node_assembly.inputs['Scale Translation'])

    return node_bio
