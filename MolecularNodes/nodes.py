import bpy
import os
import random
from typing import Optional
from . import pkg
from bpy.types import Object, GeometryNodeGroup, GeometryNodeTree
from scipy.spatial.transform import Rotation

socket_types = {
    'BOOLEAN':    'NodeSocketBool',
    'GEOMETRY':   'NodeSocketGeometry',
    'INT':        'NodeSocketInt',
    'MATERIAL':   'NodeSocketMaterial',
    'VECTOR':     'NodeSocketVector',
    'STRING':     'NodeSocketString',
    'VALUE':      'NodeSocketFloat',
    'COLLECTION': 'NodeSocketCollection',
    'TEXTURE':    'NodeSocketTexture',
    'COLOR':      'NodeSocketColor',
    'IMAGE':      'NodeSocketImage'
}


def new_node_in_group(group, kind: str, location: Optional[tuple[float, float]] = None, data_type: Optional[str] = None):
    node = group.nodes.new(kind)
    if location is not None:
        node.location = tuple(location)
    if data_type is not None:
        node.data_type = data_type
    return node


def activate_molecular_nodes_modifier_on(obj: Object):
    ''' Either get or create a MolecularNodes modifier on `obj`
        and make it the object's active modifier.
    '''
    # Does this work on multiple objects?
    # Can we make obj the active Blender object?
    obj.modifiers.active = obj.modifiers.get("MolecularNodes") \
        or obj.modifiers.new("MolecularNodes", "NODES")
    return obj.modifiers.active


def get_asset(collection, name: str, path_suffix: str):
    ''' Get or create the asset called `name`.
        If it doesn't already exist, append it to the .blend file.
    '''
    if (asset := collection.get(name)) is not None:
        return asset
    bpy.ops.wm.append(directory=os.path.join(
        pkg.ADDON_DIR, 'assets', f'node_append_file.blend/{path_suffix}'
    ), filename=name, link=False)
    return collection[name]

def mol_append_node(node_name: str):
    return get_asset(bpy.data.node_groups, node_name, 'NodeTree')

def mol_base_material(material_name: str = 'MOL_atomic_material'):
    return get_asset(bpy.data.materials, material_name, 'Material')


def gn_new_group_empty(name: str = "Geometry Nodes") -> GeometryNodeTree:
    # If a group by this name already exists, use that
    if (group := bpy.data.node_groups.get(name)) is not None:
        return group
    # Otherwise, make a new group
    group: GeometryNodeTree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group.inputs .new('NodeSocketGeometry', 'Geometry')
    group.outputs.new('NodeSocketGeometry', 'Geometry')
    input_node  = group.nodes.new('NodeGroupInput')
    output_node = group.nodes.new('NodeGroupOutput')
    output_node.is_active_output = True
    input_node .select = False
    output_node.select = False
    input_node .location.x = -200 - input_node.width
    output_node.location.x = 200
    group.links.new(output_node.inputs[0], input_node.outputs[0])
    return group


def add_custom_node_group_to_node(
    parent_group, node_name: str, location: tuple[float, float] = (0, 0), width: int = 200
) -> GeometryNodeGroup:
    mol_append_node(node_name)
    node: GeometryNodeGroup = new_node_in_group(
        parent_group, 'GeometryNodeGroup', location=location)
    node.node_tree = bpy.data.node_groups[node_name]
    node.width = width  # TODO Check that this works
    return node


def add_custom_node_group(parent_group, node_name, location=(0, 0), width=200):
    return add_custom_node_group_to_node(parent_group.node_group, node_name, location, width)


def create_starting_nodes_starfile(obj):

    node_mod = activate_molecular_nodes_modifier_on(obj)
    node_name = f"MOL_starfile_{obj.name}"

    # If there already exists a node tree by this name, use that
    if (node_group := bpy.data.node_groups.get(node_name)) is not None:
        node_mod.node_group = node_group
        return node_group

    # Create a new Geometry Nodes node group for this molecule
    node_group: GeometryNodeTree = gn_new_group_empty(node_name)
    node_mod.node_group = node_group
    node_group.inputs.new("NodeSocketObject", "Molecule")
    node_group.inputs.new("NodeSocketInt", "Image")
    node_group.inputs["Image"].default_value = 1
    node_group.inputs["Image"].min_value = 1
    node_group.inputs.new("NodeSocketBool", "Simplify")
    # Move the input and output nodes for the group
    node_input = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Input",)]
    node_input.location = (0, 0)
    node_output = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Output",)]
    node_output.location = (900, 0)

    node_delete = new_node_in_group(node_group, "GeometryNodeDeleteGeometry",
                                    location=(500, 0))

    node_instance = new_node_in_group(node_group, "GeometryNodeInstanceOnPoints",
                                      location=(675, 0))

    node_get_imageid = new_node_in_group(node_group, "GeometryNodeInputNamedAttribute",
                                         data_type='INT', location=(0, 200))
    node_get_imageid.inputs['Name'].default_value = "MOLImageId"

    node_subtract = new_node_in_group(node_group, "ShaderNodeMath",
                                      location=(160, 200))
    node_subtract.operation = "SUBTRACT"
    node_subtract.inputs[1].default_value = 1
    node_subtract.inputs[0].default_value = 1

    node_compare = new_node_in_group(node_group, "FunctionNodeCompare",
                                     data_type='INT', location=(320, 200))
    node_compare.operation = "NOT_EQUAL"

    node_object_info = new_node_in_group(node_group, "GeometryNodeObjectInfo",
                                         location=(200, -200))

    node_get_rotation = new_node_in_group(node_group, "GeometryNodeInputNamedAttribute",
                                          data_type='FLOAT_VECTOR', location=(450, -200))
    node_get_rotation.inputs['Name'].default_value = "MOLRotation"

    node_get_id = new_node_in_group(node_group, "GeometryNodeInputID",
                                    location=(0, -200))

    X = iter(range(200, 200 * 6, 200))
    Y = -400

    node_statistics = new_node_in_group(node_group, "GeometryNodeAttributeStatistic",
                                        location=(next(X), Y))

    node_compare_maxid = new_node_in_group(node_group, "FunctionNodeCompare",
                                           location=(next(X), Y))
    node_compare_maxid.operation = "EQUAL"

    node_bool_math = new_node_in_group(node_group, "FunctionNodeBooleanMath",
                                       location=(next(X), Y))
    node_bool_math.operation = "OR"

    node_switch = new_node_in_group(node_group, "GeometryNodeSwitch",
                                    location=(next(X), Y))

    node_cone = new_node_in_group(node_group, "GeometryNodeMeshCone",
                                  location=(next(X), Y))

    node_group.links.new(node_input.outputs[0],    node_delete.inputs[0])
    node_group.links.new(node_delete.outputs[0],   node_instance.inputs[0])
    node_group.links.new(node_instance.outputs[0], node_output.inputs[0])

    node_group.links.new(node_input.outputs[1], node_object_info.inputs[0])
    node_group.links.new(node_input.outputs[2], node_subtract.inputs[0])
    node_group.links.new(node_input.outputs[3], node_bool_math.inputs[0])

    node_group.links.new(node_subtract.outputs[0],             node_compare.inputs[2])
    node_group.links.new(node_get_imageid.outputs[4],          node_compare.inputs[3])
    node_group.links.new(node_compare.outputs[0],              node_delete.inputs[1])
    node_group.links.new(node_statistics.outputs[4],           node_compare_maxid.inputs[0])
    node_group.links.new(node_compare_maxid.outputs[0],        node_bool_math.inputs[1])
    node_group.links.new(node_get_id.outputs[0],               node_statistics.inputs[2])
    node_group.links.new(node_object_info.outputs["Geometry"], node_statistics.inputs[0])
    node_group.links.new(node_bool_math.outputs[0],            node_switch.inputs[1])
    node_group.links.new(node_object_info.outputs["Geometry"], node_switch.inputs[14])
    node_group.links.new(node_cone.outputs[0],                 node_switch.inputs[15])
    node_group.links.new(node_switch.outputs[6],               node_instance.inputs["Instance"])
    node_group.links.new(node_get_rotation.outputs[0],         node_instance.inputs["Rotation"])

    # Need to manually set Image input to 1, otherwise it will be 0 (even though default is 1)
    node_mod['Input_3'] = 1


def create_starting_nodes_density(obj, threshold: float = 0.8):

    node_mod = activate_molecular_nodes_modifier_on(obj)
    node_name = f"MOL_density_{obj.name}"

    # If there already exists a node tree by this name, use that
    if (node_group := bpy.data.node_groups.get(node_name)) is not None:
        node_mod.node_group = node_group
        return node_group

    # Create a new Geometry Nodes node group for this molecule
    node_group: GeometryNodeTree = gn_new_group_empty(node_name)
    node_mod.node_group = node_group
    # Move the input and output nodes for the group
    node_input = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Input",)]
    node_input.location = (0, 0)
    node_output = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Output",)]
    node_output.location = (800, 0)

    node_density = add_custom_node_group_to_node(node_mod.node_group, 'MOL_style_density_surface', (400, 0))
    node_density.inputs['Material'].default_value = mol_base_material()
    node_density.inputs['Density Threshold'].default_value = threshold

    node_group.links.new(node_input.outputs[0], node_density.inputs[0])
    node_group.links.new(node_density.outputs[0], node_output.inputs[0])


def create_starting_node_tree(obj, coll_frames, starting_style = "atoms"):

    node_mod = activate_molecular_nodes_modifier_on(obj)
    name = f"MOL_{obj.name}"

    # If a node group by this name already exists, use it and make no changes
    if (node_group := bpy.data.node_groups.get(name)) is not None:
        node_mod.node_group = node_group
        return node_group
    # Otherwise, create a new GN node group for this molecule
    node_group: GeometryNodeTree = gn_new_group_empty(name)
    node_mod.node_group = node_group

    # Move the input and output nodes for the group
    node_input = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Input",)]
    node_input.location = (0, 0)
    node_output = node_mod.node_group.nodes[bpy.app.translations.pgettext_data("Group Output",)]
    node_output.location = (800, 0)

    # node_properties = add_custom_node_group_to_node(node_group.node_group, 'MOL_prop_setup', [0, 0])
    node_colour = add_custom_node_group_to_node(node_mod.node_group, 'MOL_color_set_common', [200, 0])

    node_random_colour = new_node_in_group(node_group, "FunctionNodeRandomValue",
                                           data_type='FLOAT_VECTOR', location=(-60, -200))

    node_chain_id = new_node_in_group(node_group, "GeometryNodeInputNamedAttribute",
                                      data_type='INT', location=(-250, -450))
    node_chain_id.inputs['Name'].default_value = "chain_id"

    # create the links between the the nodes that have been established
    link = node_group.links.new
    link(node_input.outputs['Geometry'], node_colour.inputs['Atoms'])
    link(node_colour.outputs['Atoms'], node_output.inputs['Geometry'])
    link(node_random_colour.outputs['Value'], node_colour.inputs['Carbon'])
    link(node_chain_id.outputs[4], node_random_colour.inputs['ID'])

    styles = [
        'MOL_style_atoms_cycles', 'MOL_style_cartoon',
        'MOL_style_ribbon_protein', 'MOL_style_ball_and_stick',
    ]

    # BUG list indices must be integers or slices, not str
    node_style = add_custom_node_group_to_node(node_mod.node_group, styles[starting_style], location=(500, 0))
    link(node_colour.outputs['Atoms'], node_style.inputs['Atoms'])
    link(node_style.outputs[0], node_output.inputs['Geometry'])
    node_style.inputs['Material'].default_value = mol_base_material()

    if not coll_frames: return
    # If there are multiple frames, set up the nodes required for an animation
    node_output.location = (1100, 0)
    node_style.location = (800, 0)

    node_animate_frames = add_custom_node_group_to_node(node_group, 'MOL_animate_frames', [500, 0])
    node_animate_frames.inputs['Frames'].default_value = coll_frames

    # node_animate_frames.inputs['Absolute Frame Position'].default_value = True

    node_animate = add_custom_node_group_to_node(node_group, 'MOL_animate_value', [500, -300])
    link(node_colour.outputs['Atoms'], node_animate_frames.inputs['Atoms'])
    link(node_animate_frames.outputs['Atoms'], node_style.inputs['Atoms'])
    link(node_animate.outputs['Animate 0..1'], node_animate_frames.inputs['Animate 0..1'])


def create_custom_surface(name, n_chains, *, merge_kind='join_geometry'):
    # If there already exists a group by this name, use that
    if (group := bpy.data.node_groups.get(name)) is not None:
        return group

    # Get the node to create a loop from
    looping_node = mol_append_node('MOL_style_surface_single')

    # Create new empty data block
    group: GeometryNodeTree = bpy.data.node_groups.new(name, 'GeometryNodeTree')

    # loop over the inputs and create an input for each
    for i in looping_node.inputs.values():
        group_input = group.inputs.new(socket_types.get(i.type), i.name)
        try:
            group_input.default_value = i.default_value
        except AttributeError:
            pass

    group.inputs['Selection'].hide_value = True

    group.outputs.new(socket_types.get('GEOMETRY'), 'Surface Geometry')
    group.outputs.new(socket_types.get('GEOMETRY'), 'Surface Instances')

    # Add the inputs and the outputs inside of the node
    node_input  = new_node_in_group(group, 'NodeGroupInput',  location=(-300, 0))
    node_output = new_node_in_group(group, 'NodeGroupOutput', location=( 800, 0))

    node_input = group.nodes[bpy.app.translations.pgettext_data("Group Input",)]
    # node_output = group.nodes[bpy.app.translations.pgettext_data("Group Output",)]

    node_chain_id = new_node_in_group(group, "GeometryNodeInputNamedAttribute",
        data_type='INT', location=(-250, -450))
    node_chain_id.inputs['Name'].default_value = "chain_id"

    # for each chain, separate the geometry and choose only that chain, pipe through
    # a surface node and then join it all back together again
    list_node_surface = []
    height_offset = 300
    for chain in range(n_chains):
        offset = 0 - chain * height_offset
        node_separate = new_node_in_group(
            group, 'GeometryNodeSeparateGeometry', location=(120, offset))

        node_compare = new_node_in_group(group, 'FunctionNodeCompare',
                                         data_type='INT', location=(-100, offset))
        node_compare.operation = 'EQUAL'

        group.links.new(node_chain_id.outputs[4], node_compare.inputs[2])
        node_compare.inputs[3].default_value = chain
        group.links.new(node_compare.outputs['Result'], node_separate.inputs['Selection'])
        group.links.new(node_input.outputs[0], node_separate.inputs['Geometry'])

        node_surface_single = new_node_in_group(
            group, 'GeometryNodeGroup', location=(300, offset))
        node_surface_single.node_tree = looping_node

        group.links.new(node_separate.outputs['Selection'], node_surface_single.inputs['Atoms'])

        for i in node_surface_single.inputs.values():
            if i.type != 'GEOMETRY':
                group.links.new(node_input.outputs[i.name], i)

        list_node_surface.append(node_surface_single)

    # Create join geometry
    node_join_geometry        = new_node_in_group(group, 'GeometryNodeJoinGeometry',
                                                  location=(500,    0))
    node_join_volume          = new_node_in_group(group, 'GeometryNodeJoinGeometry',
                                                  location=(500, -300))
    node_geometry_to_instance = new_node_in_group(group, 'GeometryNodeGeometryToInstance',
                                                  location=(500, -600))

    # Link the nodes in reverse order
    for n in reversed(list_node_surface):
        group.links.new(n.outputs[0], node_join_geometry.inputs['Geometry'])
        group.links.new(n.outputs[0], node_geometry_to_instance.inputs['Geometry'])

    # Link the joined nodes to the outputs
    group.links.new(node_join_geometry.outputs['Geometry'], node_output.inputs[0])
    group.links.new(node_geometry_to_instance.outputs['Instances'], node_output.inputs['Surface Instances'])
    return group


def rotation_matrix(node_group, mat, location: tuple[float, float] = (0, 0), world_scale: float = 0.01) -> GeometryNodeGroup:
    """Add a Rotation & Translation node from a 3x4 matrix.

    Args:
        node_group (_type_): Parent node group to add this new node to.
        mat (_type_): 3x4 rotation & translation matrix
        location (list, optional): Position to add the node in the node tree. Defaults to [0,0].
        world_scale(float, optional): Scaling factor for the world. Defaults to 0.01.
    Returns:
        _type_: Newly created node tree.
    """
    node = new_node_in_group(node_group, 'GeometryNodeGroup', location=location)
    node.node_tree = mol_append_node('MOL_utils_rot_trans')
    # Calculate Euler angles from rotation matrix
    rotation = Rotation.from_matrix(mat[:3,:3]).as_euler('xyz')
    node.inputs[0].default_value[:3] = rotation[:3]                   # Set rotation
    node.inputs[1].default_value[:3] = mat[:3,3:][:3] * world_scale  # Set translation
    return node


def chain_selection(node_name, chain_names: list[str], attribute, start: int = 0, format=str):
    """
    Given a list of chain names, will create a node which takes an Integer input,
    and has a boolean tick box for each item in the input list.
    Returns the resulting selection and the inversion of the selection.
    Will format the resulting labels using `format` (by default, just the chain name).
    Use to construct chain selections for specific proteins.
    """
    # If the group already exists, just return it
    if (group := bpy.data.node_groups.get(node_name)) is not None:
        return group

    activate_molecular_nodes_modifier_on(bpy.context.active_object)

    # The custom node group data block, where everything will go
    chain_group = bpy.data.node_groups.new(node_name, "GeometryNodeTree")
    # The required group node input
    chain_group_in = chain_group.nodes.new("NodeGroupInput")
    chain_group_in.location = (-200, 0)
    # A named attribute node that gets the chain_number attribute
    # and is used for the later selection algebra
    chain_number_node = new_node_in_group(
        chain_group, "GeometryNodeInputNamedAttribute", data_type='INT', location=(-200, 200))
    chain_number_node.inputs[0].default_value = attribute
    chain_number_node.outputs.get('Attribute')
    # Create a boolean input for the group for each item in the list
    for chain_name in chain_names:
        # Create a boolean input for the name, and name it after the chain
        chain_group.inputs.new("NodeSocketBool", format(chain_name))
    horizontal_margin = 180  # Horizontal distance between nodes
    for i, chain_name in enumerate(chain_names):
        current_node = new_node_in_group(chain_group, "GeometryNodeGroup", location=(i * horizontal_margin, 200))
        current_node.node_tree = mol_append_node('MOL_utils_bool_chain')
        current_node.inputs['number_matched'].default_value = i + start
        # link from the the named attribute node chain_number into the other inputs
        if i == 0:
            # for some reason, you can't link with the first output of the named attribute node. Might
            # be a bug, which might be changed later, so I am just going through a range of numbers for
            # the named attribute node outputs, to link whatever it ends up being. Dodgy I know.
            # TODO revisit this and see if it is fixed and clean up code
            for j in range(5):
                try:
                    chain_group.links.new(chain_number_node.outputs[j], current_node.inputs['number_chain_in'])
                except:  # What kinds of exception are expected?
                    continue
        chain_group.links.new(chain_group_in.outputs[i], current_node.inputs['bool_include'])
        if i > 0:
            chain_group.links.new(previous_node.outputs['number_chain_out'], current_node.inputs['number_chain_in'])
            chain_group.links.new(previous_node.outputs['bool_chain_out'], current_node.inputs['bool_chain_in'])
        previous_node = current_node

    chain_group_out = new_node_in_group(chain_group, "NodeGroupOutput", location=((i + 1) * horizontal_margin, 200))
    chain_group.outputs.new("NodeSocketBool", "Selection")
    chain_group.outputs.new("NodeSocketBool", "Inverted")
    chain_group.links.new(current_node.outputs['bool_chain_out'], chain_group_out.inputs['Selection'])

    bool_math = new_node_in_group(chain_group, "FunctionNodeBooleanMath", location=(i * horizontal_margin, 50))
    bool_math.operation = "NOT"
    chain_group.links.new(current_node.outputs['bool_chain_out'], bool_math.inputs[0])
    chain_group.links.new(bool_math.outputs[0], chain_group_out.inputs['Inverted'])
    # create an empty node group group inside of the node tree
    # link the just-created custom node group data to the node group in the tree
    # new_node_group = node_mod.node_group.nodes.new("GeometryNodeGroup")
    # new_node_group.node_tree = bpy.data.node_groups[chain_group.name]
    # Slightly widen the newly created node
    # node_mod.node_group.nodes[-1].width = 200
    # the chain_id_list and output_name are passed in from the operator when it is called
    # these are custom properties that are associated with the object when it is constructed
    return chain_group


def chain_color(node_name, input_list, format=lambda s: f"Chain {s}"):
    """
    Given the input list of chain names, will create a node group which uses
    the chain_id named attribute to manually set the colours for each of the chains.
    """
    activate_molecular_nodes_modifier_on(bpy.context.active_object)

    # The custom node group data block, where everything will go
    chain_group = bpy.data.node_groups.new(node_name, "GeometryNodeTree")
    # The required group node input
    node_input = new_node_in_group(chain_group, "NodeGroupInput", location=(-200, 0))

    # A named attribute node that gets the chain_number attribute
    # and is used for the later selection algebra
    chain_number_node = new_node_in_group(
        chain_group, "GeometryNodeInputNamedAttribute",
        data_type='INT', location=(-200, 400))
    chain_number_node.inputs[0].default_value = 'chain_id'
    chain_number_node.outputs.get('Attribute')

    horizontal_margin = 180  # Horizontal distance between nodes
    for i, chain_name in enumerate(input_list):
        x = i * horizontal_margin
        current_chain = format(chain_name)
        # Node compare inputs 2 & 3
        node_compare = new_node_in_group(chain_group, 'FunctionNodeCompare',
                                         data_type='INT', location=(x, 100))
        node_compare.operation = 'EQUAL'

        node_compare.inputs[3].default_value = i

        # Link the named attribute to the compare
        chain_group.links.new(chain_number_node.outputs[4], node_compare.inputs[2])

        node_color = new_node_in_group(chain_group, 'GeometryNodeSwitch',
                                       location=(x, -100))
        node_color.input_type = 'RGBA'

        # Create an input for this chain
        chain_group.inputs.new("NodeSocketColor", current_chain)
        chain_group.inputs[current_chain].default_value = [random.random(), random.random(), random.random(), 1]
        # Switch input colours 10 and 11
        chain_group.links.new(node_input.outputs[current_chain], node_color.inputs[11])
        chain_group.links.new(node_compare.outputs['Result'], node_color.inputs['Switch'])

        if i > 0:
            chain_group.links.new(node_color_previous.outputs[4], node_color.inputs[10])

        node_color_previous = node_color

    chain_group.outputs.new("NodeSocketColor", "Color")
    node_output = new_node_in_group(chain_group, "NodeGroupOutput", location=(x, 200))
    chain_group.links.new(node_color.outputs[4], node_output.inputs['Color'])
    return chain_group


def resid_multiple_selection(node_name: str, input_resid_string: str):
    """
    Returns a node group that takes an integer input and creates a boolean
    tick box for each item in the input list. Outputs are the selected
    residues and the inverse selection. Used for constructing chain
    selections in specific proteins.
    """
    # Preprocess input_resid_string to allow fuzzy matching
    for k, v in ({k: ',' for k in ';/+ .'} | {k: '-' for k in '_=:'}).items():
        input_resid_string = input_resid_string.replace(k, v)

    # Parse input_resid_string into sub selecting string list
    sub_list = list(filter(bool, input_resid_string.split(',')))

    vertical_margin = 100  # Vertical distance between nodes

    activate_molecular_nodes_modifier_on(bpy.context.active_object)

    # Custom node group data block, where everything will go
    residue_id_group = bpy.data.node_groups.new(node_name, "GeometryNodeTree")
    # Required group node input
    residue_id_group_in = new_node_in_group(
        residue_id_group, "NodeGroupInput",
        location=(0, -vertical_margin * len(sub_list) / 2))

    for residue_id in sub_list:

        if '-' in residue_id:
            resid_start, resid_end, *_ = residue_id.split('-')
            # Set two new inputs
            residue_id_group.inputs.new("NodeSocketInt", 'res_id_start').default_value = int(resid_start)
            residue_id_group.inputs.new("NodeSocketInt", 'res_id_end').default_value = int(resid_end)
        else:
            # Set a new input and set the resid
            residue_id_group.inputs.new("NodeSocketInt", 'res_id').default_value = int(residue_id)

    num_new_links: int = 0  # A counter for MOL_sel_res_id* nodes
    for i, residue_id in enumerate(sub_list):

        # Add a bool_math block
        bool_math = new_node_in_group(residue_id_group, "FunctionNodeBooleanMath",
                                      location=(400, -(i + 1) * vertical_margin))
        bool_math.operation = "OR"

        # Add a new node of MOL_sel_res_id or MOL_sek_res_id_range
        current_node = residue_id_group.nodes.new("GeometryNodeGroup")

        if '-' in residue_id:
            # A residue range
            current_node.node_tree = mol_append_node('MOL_sel_res_id_range')
            residue_id_group.links.new(residue_id_group_in.outputs[num_new_links], current_node.inputs[0])
            num_new_links += 1
            residue_id_group.links.new(residue_id_group_in.outputs[num_new_links], current_node.inputs[1])
            num_new_links += 1
        else:
            # Create a node
            current_node.node_tree = mol_append_node('MOL_sel_res_id')
            # Link the input of MOL_sel_res_id
            residue_id_group.links.new(residue_id_group_in.outputs[num_new_links], current_node.inputs[0])
            num_new_links += 1

        # Set the coordinates
        current_node.location = (200, -(i + 1) * vertical_margin)

        if i == 0:
            # link the first residue selection to the first input of its OR block
            residue_id_group.links.new(current_node.outputs['Selection'],bool_math.inputs[0])
        else:
            # if it is not the first residue selection, link the output to the previous or block
            residue_id_group.links.new(current_node.outputs['Selection'], previous_bool_node.inputs[1])
            # link the ouput of previous OR block to the current OR block
            residue_id_group.links.new(previous_bool_node.outputs[0], bool_math.inputs[0])

        previous_bool_node = bool_math

    # Add output block
    residue_id_group_out = new_node_in_group(residue_id_group, "NodeGroupOutput",
                                             location=(800, -(i + 1) / 2 * vertical_margin))
    residue_id_group.outputs.new("NodeSocketBool", "Selection")
    residue_id_group.outputs.new("NodeSocketBool", "Inverted")
    residue_id_group.links.new(previous_bool_node.outputs[0], residue_id_group_out.inputs['Selection'])
    invert_bool_math = new_node_in_group(residue_id_group, "FunctionNodeBooleanMath",
                                         location=(600, -(i + 1) / 3 * 2 * vertical_margin))
    invert_bool_math.operation = "NOT"
    residue_id_group.links.new(previous_bool_node.outputs[0], invert_bool_math.inputs[0])
    residue_id_group.links.new(invert_bool_math.outputs[0], residue_id_group_out.inputs['Inverted'])
    return residue_id_group
