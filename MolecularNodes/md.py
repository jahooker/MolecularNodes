"""
Importing molecular dynamics trajectories and associated files.
"""

__name__ = "MolecularNodes.trajectory"
__author__ = "Brady Johnston"

import bpy
import numpy as np
import warnings
import MDAnalysis as mda
import data
import coll
import obj
import nodes
from load import att_atomic_number


bpy.types.Scene.mol_import_md_topology = bpy.props.StringProperty(
    name='path_topology',
    description='File path for the toplogy file for the trajectory',
    options={'TEXTEDIT_UPDATE'}, default='', subtype='FILE_PATH', maxlen=0)
bpy.types.Scene.mol_import_md_trajectory = bpy.props.StringProperty(
    name='path_trajectory',
    description='File path for the trajectory file for the trajectory',
    options={'TEXTEDIT_UPDATE'}, default='', subtype='FILE_PATH', maxlen=0)
bpy.types.Scene.mol_import_md_name = bpy.props.StringProperty(
    name='mol_md_name',
    description='Name of the molecule on import',
    options={'TEXTEDIT_UPDATE'}, default='NewTrajectory', subtype='NONE', maxlen=0)
bpy.types.Scene.mol_import_md_frame_start = bpy.props.IntProperty(
    name="mol_import_md_frame_start",
    description="Frame start for importing MD trajectory",
    subtype='NONE', default=0)
bpy.types.Scene.mol_import_md_frame_step = bpy.props.IntProperty(
    name="mol_import_md_frame_step",
    description="Frame step for importing MD trajectory",
    subtype='NONE', default=1)
bpy.types.Scene.mol_import_md_frame_end = bpy.props.IntProperty(
    name="mol_import_md_frame_end",
    description="Frame end for importing MD trajectory",
    subtype='NONE', default=49)
bpy.types.Scene.mol_md_selection = bpy.props.StringProperty(
    name='md_selection',
    description='Custom selection string when importing MD simulation. '
                'See: "https://docs.mdanalysis.org/stable/documentation_pages/selections.html"',
    options={'TEXTEDIT_UPDATE'}, default='not (name H* or name OW)', subtype='NONE')
bpy.types.Scene.list_index = bpy.props.IntProperty(
    name="Index for trajectory selection list.", default=0)


class MOL_OT_Import_Protein_MD(bpy.types.Operator):

    bl_idname = "mol.import_protein_md"
    bl_label = "Import Protein MD"
    bl_description = "Load molecular dynamics trajectory"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        file_top          = bpy.context.scene.mol_import_md_topology
        file_traj         = bpy.context.scene.mol_import_md_trajectory
        name              = bpy.context.scene.mol_import_md_name
        selection         = bpy.context.scene.mol_md_selection
        md_start          = bpy.context.scene.mol_import_md_frame_start
        md_step           = bpy.context.scene.mol_import_md_frame_step
        md_end            = bpy.context.scene.mol_import_md_frame_end
        include_bonds     = bpy.context.scene.mol_import_include_bonds
        custom_selections = bpy.context.scene.trajectory_selection_list

        mol_object, coll_frames = load_trajectory(
            file_top=file_top, file_traj=file_traj,
            md_start=md_start, md_end=md_end, md_step=md_step,
            name=name, selection=selection, include_bonds=include_bonds,
            custom_selections=custom_selections)
        n_frames = len(coll_frames.objects)

        nodes.create_starting_node_tree(
            obj=mol_object, coll_frames=coll_frames,
            starting_style=bpy.context.scene.mol_import_default_style)
        bpy.context.view_layer.objects.active = mol_object
        self.report({'INFO'}, message=f"Imported {repr(file_top)} as {mol_object.name} "
                                      f"with {n_frames} frames from {repr(file_traj)}.")
        return {"FINISHED"}


def get_bonds(univ, selection: str) -> np.ndarray:

    if not hasattr(univ, 'bonds'):
        return np.array(())

    if not selection:
        return univ.bonds.indices

    # Reindex bonds
    index_map = {index: i for i, index in enumerate(univ.atoms.indices)}
    # univ.atoms.indices should have no two elements the same
    reindexed_bonds = []
    for bond in univ.bonds.indices:
        try:
            reindexed_bonds.append([index_map[j] for j in bond])
        except KeyError:
            pass  # Ignore bonds involving deselected atoms.
    return np.array(reindexed_bonds)


def att_vdw_radii(univ, elements, world_scale: float):
    try:
        vdw_radii = np.array([mda.topology.tables.vdwradii.get(x, 1)
                                for x in np.char.upper(elements)])
    except:
        # if fail to get radii, just return radii of 1 for everything as a backup
        vdw_radii = np.ones(len(univ.atoms.names))
        warnings.warn("Unable to extract VDW Radii. Defaulting to 1 for all points.")
    return vdw_radii * world_scale


def att_res_id(univ):
    return univ.atoms.resnums


def att_res_name(univ):
    res_names =  np.array([x[:3] for x in univ.atoms.resnames])
    return np.array([
        data.residues[x]['res_name_num'] if x in data.residues else 0
        for x in res_names])


def att_b_factor(univ):
    return univ.atoms.tempfactors


def att_chain_id(univ, mol_object):
    chain_id = univ.atoms.chainIDs
    chain_id_unique = np.unique(chain_id)
    chain_id_num = np.array([np.where(x == chain_id_unique)[0][0] for x in chain_id])
    mol_object['chain_id_unique'] = chain_id_unique
    return chain_id_num


def bool_selection(univ, selection):
    # For each atom, is it in the selection?
    return np.isin(univ.atoms.ix, univ.select_atoms(selection).ix).astype(bool)


def att_is_backbone(univ):
    return bool_selection(univ, "backbone or nucleicbackbone")


def att_is_alpha_carbon(univ):
    return bool_selection(univ, 'name CA')


def att_is_solvent(univ):
    return bool_selection(univ, 'name OW or name HW1 or name HW2')


def att_atom_type(univ):
    return np.array(univ.atoms.types, dtype=int)


def att_is_nucleic(univ):
    return bool_selection(univ, 'nucleic')


def att_is_peptide(univ):
    return bool_selection(univ, 'protein')


def load_trajectory(
    file_top, file_traj, name: str = "NewTrajectory", md_start: int = 0, md_end: int = 49,
    md_step: int = 1, world_scale: float =0.01, include_bonds: bool = True,
    selection: str = "not (name H* or name OW)", custom_selections=None) -> tuple[bpy.types.Object, bpy.types.Collection]:
    """
    Loads a molecular dynamics trajectory from the specified files.

    Parameters:
    ----------
    file_top : str
        The path to the topology file.
    file_traj : str
        The path to the trajectory file.
    name : str, optional
        The name of the trajectory (default: "default").
    md_start : int, optional
        The starting frame of the trajectory to load (default: 0).
    md_end : int, optional
        The ending frame of the trajectory to load (default: 49).
    md_step : int, optional
        The step size between frames to load (default: 1).
    world_scale : float, optional
        The scaling factor for the world coordinates (default: 0.01).
    include_bonds : bool, optional
        Whether to include bond information if available (default: True).
    selection : str, optional
        The selection string for atom filtering (default: "not (name H* or name OW)").
        Uses MDAnalysis selection syntax.
    custom_selections : dict or None, optional
        A dictionary of custom selections for atom filtering with
        {'name' : 'selection string'} (default: None).

    Returns:
    -------
    mol_object : bpy.types.Object
        The loaded topology file as a blender object.
    coll_frames : bpy.types.Collection
        The loaded trajectory as a blender collection.

    Raises:
    ------
    FileNotFoundError
        If the topology or trajectory file is not found.
    IOError
        If there is an error reading the files.
    """
    # Load the trajectory
    univ = mda.Universe(file_top, file_traj) if file_traj else mda.Universe(file_top)

    # Separate the trajectory, separate to the topology or the subsequence selections
    traj = univ.trajectory[md_start:md_end:md_step]

    # If there is a non-blank selection, apply the selection text to the universe for later use.
    # This also affects the trajectory, even though it has been separated earlier.
    if selection:
        try:
            univ = univ.select_atoms(selection)
        except:
            warnings.warn(f"Unable to apply selection: '{selection}'. Loading entire topology.")

    # Try to extract the elements from the topology.
    try:
        elements = univ.atoms.elements.tolist()
    # If the universe doesn't contain the element information, 
    # then guess based on the atom names in the topology.
    except:
        try:
            elements = [mda.topology.guessers.guess_atom_element(x) for x in univ.atoms.names]
        except:
            pass

    bonds = get_bonds(univ, selection) if include_bonds else np.array(())

    # Create the initial model
    mol_object = obj.create_object(
        name=name, collection=coll.mn(),
        locations=univ.atoms.positions * world_scale, bonds=bonds)

    ## Add the attributes for the model

    # The attributes for the model are initially defined as single-use functions.
    # This allows for a loop that attempts to add each attibute by calling the function. 
    # Only during this loop will the call fail if the attribute isn't accessible, 
    # and the warning is reported there rather than setting up a try: except: for each individual attribute
    # which makes some really messy code.

    attributes = (
        {'name': 'atomic_number',   'value': lambda: att_atomic_number(elements),                'type': 'INT',     'domain': 'POINT'},
        {'name': 'vdw_radii',       'value': lambda: att_vdw_radii(univ, elements, world_scale), 'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'res_id',          'value': lambda: att_res_id(univ),                           'type': 'INT',     'domain': 'POINT'},
        {'name': 'res_name',        'value': lambda: att_res_name(univ),                         'type': 'INT',     'domain': 'POINT'},
        {'name': 'b_factor',        'value': lambda: att_b_factor(univ),                         'type': 'float',   'domain': 'POINT'},
        {'name': 'chain_id',        'value': lambda: att_chain_id(univ, mol_object),             'type': 'INT',     'domain': 'POINT'},
        {'name': 'atom_types',      'value': lambda: att_atom_type(univ),                        'type': 'INT',     'domain': 'POINT'},
        {'name': 'is_backbone',     'value': lambda: att_is_backbone(univ),                      'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_alpha_carbon', 'value': lambda: att_is_alpha_carbon(univ),                  'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_solvent',      'value': lambda: att_is_solvent(univ),                       'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_nucleic',      'value': lambda: att_is_nucleic(univ),                       'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_peptide',      'value': lambda: att_is_peptide(univ),                       'type': 'BOOLEAN', 'domain': 'POINT'},
    )

    for att in attributes:
        # tries to add the attribute to the mesh by calling the 'value' function
        # which returns the required values to be added to the domain.
        try:
            obj.add_attribute(mol_object, att['name'], att['value'](), att['type'], att['domain'])
        except:
            warnings.warn(f"Unable to add attribute: {att['name']}.")

    # add the custom selections if they exist
    if custom_selections:
        for sel in custom_selections:
            try:
                obj.add_attribute(
                    obj=mol_object, name=sel.name, 
                    data=bool_selection(sel.selection), data_type="BOOLEAN",
                    domain="POINT")
            except:
                warnings.warn(f"Unable to add custom selection: {sel.name}")

    coll_frames = coll.frames(name)

    add_occupancy = True
    for ts in traj:
        frame = obj.create_object(
            name=f"{name}_frame_{ts.frame}", collection=coll_frames,
            locations=univ.atoms.positions * world_scale)
        # adds occupancy data to each frame if it exists
        # This is mostly for people who want to store frame-specific information in the
        # b_factor but currently neither biotite nor MDAnalysis give access to frame-specific
        # b_factor information. MDAnalysis gives frame-specific access to the `occupancy`
        # so currently this is the only method to get frame-specific data into MN
        # for more details: https://github.com/BradyAJohnston/MolecularNodes/issues/128
        if add_occupancy:
            try:
                obj.add_attribute(frame, 'occupancy', ts.data['occupancy'])
            except:
                add_occupancy = False

    # disable the frames collection from the viewer
    bpy.context.view_layer.layer_collection.children[coll.mn().name].children[coll_frames.name].exclude = True

    return mol_object, coll_frames


#### UI

def bpy_register(cls: type) -> type:
    bpy.utils.register_class(cls)
    return cls


@bpy_register  # Otherwise the PropertyGroup registration fails
class TrajectorySelectionItem(bpy.types.PropertyGroup):
    """Group of properties for custom selections for MDAnalysis import."""
    bl_idname = "testing"

    name: bpy.props.StringProperty(
        name="Attribute Name",
        description="Attribute",
        default="custom_selection")

    selection: bpy.props.StringProperty(
        name="Selection String",
        description="String that provides a selection through MDAnalysis",
        default="name CA")


bpy.types.Scene.trajectory_selection_list \
    = bpy.props.CollectionProperty(type=TrajectorySelectionItem)


class MOL_UL_TrajectorySelectionListUI(bpy.types.UIList):
    """UI List"""

    def draw_item(self, context, layout, data, item,
                  icon, active_data, active_propname, index):
        custom_icon = "VIS_SEL_11"
        match self.layout_type:
            case 'DEFAULT' | 'COMPACT':
                layout.label(text=item.name, icon=custom_icon)
            case 'GRID':
                layout.alignment = 'CENTER'
                layout.label(text='', icon=custom_icon)


class TrajectorySelection_OT_NewItem(bpy.types.Operator):
    """Add a new custom selection to the list."""

    bl_idname = "trajectory_selection_list.new_item"
    bl_label = "+"

    def execute(self, context):
        context.scene.trajectory_selection_list.add()
        return {'FINISHED'}


class TrajectorySelection_OT_DeleteIem(bpy.types.Operator):

    bl_idname = "trajectory_selection_list.delete_item"
    bl_label = "-"

    @classmethod
    def poll(cls, context):
        return context.scene.trajectory_selection_list

    def execute(self, context):
        i: int = context.scene.list_index
        context.scene.trajectory_selection_list.remove(i)
        context.scene.list_index = min(max(0, i - 1), len(context.scene.trajectory_selection_list) - 1)
        return {'FINISHED'}


def panel(layout_function, scene):

    col_main = layout_function.column(heading='', align=False)
    col_main.alert = False
    col_main.enabled = True
    col_main.active = True
    col_main.label(text="Import Molecular Dynamics Trajectories")

    row_import = col_main.row()
    row_import.prop(
        bpy.context.scene, 'mol_import_md_name', 
        text="Name", emboss=True)
    row_import.operator('mol.import_protein_md', 
                        text="Load", icon='FILE_TICK')

    row_topology = col_main.row(align=True)
    row_topology.prop(
        bpy.context.scene, 'mol_import_md_topology', 
        text='Topology', emboss=True)

    row_trajectory = col_main.row()
    row_trajectory.prop(
        bpy.context.scene, 'mol_import_md_trajectory',
        text='Trajectory', icon_value=0, emboss=True)

    row_frame = col_main.row(heading='Frames', align=True)
    row_frame.prop(
        bpy.context.scene, 'mol_import_md_frame_start',
        text='Start', emboss=True)
    row_frame.prop(
        bpy.context.scene, 'mol_import_md_frame_step',
        text='Step', emboss=True)
    row_frame.prop(
        bpy.context.scene, 'mol_import_md_frame_end',
        text='End', emboss=True)
    col_main.prop(
        bpy.context.scene, 'mol_md_selection',
        text='Import Filter', emboss=True)
    col_main.separator()
    col_main.label(text='Custom Selections')
    row = col_main.row(align=True)

    row = row.split(factor=0.9)
    row.template_list('MOL_UL_TrajectorySelectionListUI', 'A list', scene,
                      'trajectory_selection_list', scene, 'list_index', rows=3)
    col = row.column()
    col.operator('trajectory_selection_list.new_item', icon='ADD', text='')
    col.operator('trajectory_selection_list.delete_item', icon='REMOVE', text='')

    if scene.list_index >= 0 and scene.trajectory_selection_list:
        col = col_main.column(align=False)
        col.separator()
        item = scene.trajectory_selection_list[scene.list_index]
        col.prop(item, 'name')
        col.prop(item, 'selection')
