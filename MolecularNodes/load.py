import os
import bpy
import numpy as np
from . import coll
import warnings
from . import data
from . import assembly
from . import nodes
from . import obj
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.structure import spread_residue_wise, annotate_sse
from biotite import InvalidFileError


bpy.types.Scene.mol_pdb_code = bpy.props.StringProperty(
    name='pdb_code', description='The 4-character PDB code to download',
    options={'TEXTEDIT_UPDATE'}, default='1bna', subtype='NONE', maxlen=4
)

bpy.types.Scene.mol_import_center = bpy.props.BoolProperty(
    name="mol_import_centre", description="Move the imported Molecule on the World Origin",
    default=False
)

bpy.types.Scene.mol_import_del_solvent = bpy.props.BoolProperty(
    name="mol_import_del_solvent", description="Delete the solvent from the structure on import",
    default=True
)

bpy.types.Scene.mol_import_include_bonds = bpy.props.BoolProperty(
    name="mol_import_include_bonds", description="Include bonds in the imported structure.",
    default=True
)

bpy.types.Scene.mol_import_panel_selection = bpy.props.IntProperty(
    name="mol_import_panel_selection", description="Import Panel Selection",
    subtype='NONE', default=0
)

bpy.types.Scene.mol_import_local_path = bpy.props.StringProperty(
    name='path_pdb', description='File path of the structure to open',
    options={'TEXTEDIT_UPDATE'}, default='', subtype='FILE_PATH', maxlen=0
)

bpy.types.Scene.mol_import_local_name = bpy.props.StringProperty(
    name='mol_name', description='Name of the molecule on import',
    options={'TEXTEDIT_UPDATE'}, default='NewMolecule', subtype='NONE', maxlen=0
)

bpy.types.Scene.mol_import_default_style = bpy.props.IntProperty(
    name="mol_import_default_style", description="Default style for importing molecules.",
    subtype='NONE', default=0
)


def molecule_rcsb(
    pdb_code: str, center_molecule=False, del_solvent=True, include_bonds=True,
    starting_style=0, setup_nodes=True
):

    mol, file = open_structure_rcsb(pdb_code=pdb_code, include_bonds=include_bonds)

    mol_object, coll_frames = create_molecule(
        mol_array=mol, mol_name=pdb_code, file=file, calculate_ss=False,
        center_molecule=center_molecule, del_solvent=del_solvent, include_bonds=include_bonds)

    if setup_nodes:
        nodes.create_starting_node_tree(
            obj=mol_object, coll_frames=coll_frames, starting_style=starting_style)

    mol_object['bio_transform_dict'] = file['bioAssemblyList']
    return mol_object


def molecule_local(
    file_path: str, mol_name="Name", include_bonds=True, center_molecule=False,
    del_solvent=True, default_style=0, setup_nodes=True
):

    file_path = os.path.abspath(file_path)
    file_ext = os.path.splitext(file_path)[1]

    if file_ext == '.pdb':
        mol, file = open_structure_local_pdb(file_path, include_bonds)
        transforms = list(assembly.get_transformations_pdb(file))
    elif file_ext == '.pdbx' or file_ext == '.cif':
        mol, file = open_structure_local_pdbx(file_path, include_bonds)
        try:
            transforms = assembly.get_transformations_pdbx(file)
        except:
            transforms = None
            # self.report({"WARNING"}, message='Unable to parse biological assembly information.')
    else:
        warnings.warn("Unable to open local file. Format not supported.")

    # If include_bonds is chosen but no bonds currently exist (mol.bonds is None),
    # then attempt to find bonds by distance.
    if include_bonds and not mol.bonds:
        mol.bonds = struc.connect_via_distances(mol[0], inter_residue=True)

    if file_ext != '.pdb' or file.get_model_count() <= 1:
        file = None

    mol_object, coll_frames = create_molecule(
        mol_array=mol, mol_name=mol_name, file=file, calculate_ss=True,
        center_molecule=center_molecule, del_solvent=del_solvent,
        include_bonds=include_bonds)

    # setup the required initial node tree on the object
    if setup_nodes:
        nodes.create_starting_node_tree(
            obj=mol_object, coll_frames=coll_frames, starting_style=default_style)

    # if transforms:
        # mol_object['bio_transform_dict'] = (transforms)
        # mol_object['bio_transnform_dict'] = 'testing'

    return mol_object


def open_structure_rcsb(pdb_code: str, include_bonds=True):
    """
    Returns a numpy array stack, where each array in the stack is a model in the the file.
    The stack will be of length 1 if there is only one model in the file.
    """
    file = mmtf.MMTFFile.read(rcsb.fetch(pdb_code, "mmtf"))
    mol = mmtf.get_structure(file, extra_fields=["b_factor", "charge"], include_bonds=include_bonds)
    return mol, file


def open_structure_local_pdb(file_path: str, include_bonds=True):
    """
    Returns a numpy array stack, where each array in the stack is a model in the the file.
    The stack will be of length 1 if there is only one model in the file
    """
    file = pdb.PDBFile.read(file_path)
    mol = pdb.get_structure(file, extra_fields=['b_factor', 'charge'], include_bonds=include_bonds)
    return mol, file


def open_structure_local_pdbx(file_path, include_bonds=True):
    """
    Returns a numpy array stack, where each array in the stack is a model in the the file.
    The stack will be of length 1 if there is only one model in the file.
    """
    file = pdbx.PDBxFile.read(file_path)
    try:
        mol = pdbx.get_structure(file, extra_fields=['b_factor', 'charge'])
    except InvalidFileError:
        mol = pdbx.get_component(file)
    # Apparently, pdbx doesn't include bond information,
    # so if requested, manually create them here.
    if include_bonds and not mol.bonds:
        mol[0].bonds = struc.bonds.connect_via_residue_names(mol[0], inter_residue=True)
    return mol, file


def pdb_get_b_factors(file):
    """
    Generate the numpy array for each model containing the b-factors.
    """
    for i in range(1, file.get_model_count() + 1):
        yield file.get_structure(model=i, extra_fields=['b_factor']).b_factor


def get_secondary_structure(mol_array, file) -> np.ndarray[int]:
    """
    Gets the secondary structure annotation that is included in mmtf files and returns it as a numerical numpy array.

    Parameters:
    -----------
    mol_array : numpy.ndarray
        The molecular coordinates array, from mmtf.get_structure()
    file : mmtf.MMTFFile
        The MMTF file containing the secondary structure information, from mmtf.MMTFFile.read()

    Returns:
    --------
    atom_sse : numpy.ndarray[int]
        Numerical numpy array representing the secondary structure of the molecule.

    Description:
    ------------
    This function uses the biotite.structure package to extract the secondary structure information from the MMTF file.
    The resulting secondary structures are `1: Alpha Helix, 2: Beta-sheet, 3: loop`.
    """

    sec_struct_codes = {-1: 'X'} | {i: item for i, item in enumerate("ISHEGBTC")}

    dssp_to_abc = {
        'X' : 0, 'I' : 1, 'S' : 3,
        'H' : 1, 'E' : 2, 'G' : 1,
        'B' : 2, 'T' : 3, 'C' : 3,
    }  # {1: 'a', 2: 'b', 3: 'c'}

    try:
        sse = file["secStructList"]
    except KeyError:
        ss_int = np.full(len(mol_array), 3)
        print('Warning: "secStructList" field missing from MMTF file. '
              'Defaulting to "loop" for all residues.')
    else:
        ss_int = np.array([dssp_to_abc[sec_struct_codes[ss]] for ss in sse], dtype=int)
    return spread_residue_wise(mol_array, ss_int)


def comp_secondary_structure(mol_array: MolecularArray):
    """Use dihedrals to compute the secondary structure of proteins

    Through biotite built-in method derivated from P-SEA algorithm (Labesse 1997)
    Returns an array with secondary structure for each atoms where:
    - 0 = '' = non-protein or not assigned by biotite annotate_sse
    - 1 = a = alpha helix
    - 2 = b = beta sheet
    - 3 = c = coil

    Inspired from https://www.biotite-python.org/examples/gallery/structure/transketolase_sse.html
    """
    # TODO Port [PyDSSP](https://github.com/ShintaroMinami/PyDSSP)
    # TODO Read 'secStructList' field from mmtf files
    conv_sse_char_int = {'': 0, 'a': 1, 'b': 2, 'c': 3}
    char_sse = annotate_sse(mol_array)
    int_sse = np.array([conv_sse_char_int[char] for char in char_sse], dtype=int)
    return spread_residue_wise(mol_array, int_sse)


MolecularArray: type = struc.AtomArray | struc.AtomArrayStack

def att_atomic_number(elements: np.ndarray[str]):
    return np.array([
        data.elements[symbol]['atomic_number'] if symbol in data.elements else -1
        for symbol in np.char.title(elements)
    ])

def att_res_id(mol_array: MolecularArray):
    return mol_array.res_id

def att_res_name(mol_array: MolecularArray, mol_object):
    id_counter = -1
    res_names = mol_array.res_name
    res_ids = mol_array.res_id
    res_nums: list[int] = []
    other_res: list[str] = []

    for i, name in enumerate(res_names):

        if name in data.residues:
            res_nums.append(data.residues[name]['res_name_num'])
        else:
            if res_names[i - 1] != name or res_ids[i] != res_ids[i - 1]:
                id_counter += 1

            unique_res_name = f'{id_counter + 100}_{name}'
            other_res.append(unique_res_name)

            n: int = np.where(np.isin(np.unique(other_res), unique_res_name))[0][0] + 100
            res_nums.append(n)

    mol_object['ligands'] = np.unique(other_res)
    return np.array(res_nums)


def att_chain_id(mol_array: MolecularArray):
    return np.searchsorted(np.unique(mol_array.chain_id), mol_array.chain_id)

def att_b_factor(mol_array: MolecularArray):
    return mol_array.b_factor

def att_vdw_radii(mol_array: MolecularArray, world_scale):
    return world_scale * np.array([
        # All coordinates are in Angstroms, so divide by 100 to convert from picometers.
        data.elements[symbol]['vdw_radii'] / 100 if symbol in data.elements else 1,
        for symbol in np.char.title(mol_array.element)
    ])

def att_atom_name(mol_array: MolecularArray):
    return np.array([data.atom_names.get(x, 9999) for x in mol_array.atom_name])

def att_lipophobicity(mol_array: MolecularArray):
    return np.array([
        data.lipophobicity[x][y] if x in data.lipophobicity
        and y in data.lipophobicity[x] else 0
        for x, y in zip(mol_array.res_name, mol_array.atom_name)
    ])

def att_charge(mol_array: MolecularArray):
    return np.array([
        data.atom_charge[x][y] if x in data.atom_charge
        and y in data.atom_charge[x] else 0
        for x, y in zip(mol_array.res_name, mol_array.atom_name)
    ])

def att_is_alpha(mol_array: MolecularArray):
    return np.isin(mol_array.atom_name, 'CA')

def att_is_solvent(mol_array: MolecularArray):
    return struc.filter_solvent(mol_array)

def att_is_backbone(mol_array: MolecularArray):
    """
    Get the atoms that appear in peptide backbone or nucleic acid phosphate backbones.
    Filter differs from the Biotite's `struc.filter_peptide_backbone()` in that this
    includes the peptide backbone oxygen atom, which biotite excludes. Additionally
    this selection also includes all of the atoms from the ribose in nucleic acids,
    and the other phosphate oxygens.
    """
    backbone_atom_names = [
        'N', 'C', 'CA', 'O',                         # peptide backbone atoms
        'P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'', # 'continuous' nucleic backbone atoms
        'O1P', 'OP1', 'O2P', 'OP2',                  # alternative names for phosphate O's
        'O4\'', 'C1\'', 'C2\'', 'O2\''               # remaining ribose atoms
    ]
    return np.isin(mol_array.atom_name, backbone_atom_names) \
        & ~struc.filter_solvent(mol_array)

def att_is_nucleic(mol_array: MolecularArray):
    return struc.filter_nucleotides(mol_array)

def att_is_peptide(mol_array: MolecularArray) -> np.ndarray[bool]:
    return struc.filter_amino_acids(mol_array) \
        | struc.filter_canonical_amino_acids(mol_array)

def att_is_hetero(mol_array: MolecularArray):
    return mol_array.hetero

def att_is_carb(mol_array: MolecularArray) -> np.ndarray[bool]:
    return struc.filter_carbohydrates(mol_array)

def att_sec_struct(mol_array: MolecularArray, file, calculate_ss):
    return comp_secondary_structure(mol_array) if calculate_ss or not file \
        else get_secondary_structure(mol_array, file)


def create_molecule(
    mol_array: MolecularArray, mol_name, center_molecule=False, file=None,
    calculate_ss=False, del_solvent=False, include_bonds=False, collection=None
):
    mol_frames: list = []
    if isinstance(mol_array, struc.AtomArrayStack):
        if mol_array.stack_depth() > 1:
            mol_frames = mol_array
        mol_array = mol_array[0]

    # Perhaps remove the solvent from the structure.
    if del_solvent:
        mol_array = mol_array[np.invert(struc.filter_solvent(mol_array))]

    world_scale = 0.01
    locations = mol_array.coord * world_scale

    if center_molecule:
        # Subtract the centroid from all of the positions,
        # to localise the molecule on the world origin.
        centroid = struc.centroid(mol_array) * world_scale
        locations -= centroid
    else:
        centroid = np.array([0, 0, 0])

    if collection is None:
        collection = coll.mn()

    bonds = []
    bond_idx = []
    if include_bonds and mol_array.bonds:
        bonds = mol_array.bonds.as_array()
        bond_idx = bonds[:, [0, 1]]
        bond_types = bonds[:, 2].copy(order='C') # the .copy(order='C') is to fix a weird ordering issue with the resulting array

    mol_object = obj.create_object(name=mol_name, collection=collection,
                                   locations=locations, bonds=bond_idx)

    # The attributes for the model are initially defined as single-use functions. This allows
    # for a loop that attempts to add each attibute by calling the function. Only during this
    # loop will the call fail if the attribute isn't accessible, and the warning is reported
    # there rather than setting up a try: except: for each individual attribute which makes
    # some really messy code.

    # I still don't like this as an implementation,
    # and welcome any cleaner approaches that anybody might have.

    # Add information about the bond types to the model on the edge domain
    # Bond types: 'ANY' = 0, 'SINGLE' = 1, 'DOUBLE' = 2, 'TRIPLE' = 3, 'QUADRUPLE' = 4
    # 'AROMATIC_SINGLE' = 5, 'AROMATIC_DOUBLE' = 6, 'AROMATIC_TRIPLE' = 7
    # https://www.biotite-python.org/apidoc/biotite.structure.BondType.html#biotite.structure.BondType
    if include_bonds:
        try:
            obj.add_attribute(
                obj=mol_object, name='bond_type', data=bond_types,
                date_type='INT', domain='EDGE'
            )
        except:
            warnings.warn('Unable to add bond types to the molecule.')

    # These attributes will be added to the structure.
    attributes = (
        {'name': 'res_id',          'value': lambda: att_res_id(mol_array),                         'type': 'INT',     'domain': 'POINT'},
        {'name': 'res_name',        'value': lambda: att_res_name(mol_array, mol_object),           'type': 'INT',     'domain': 'POINT'},
        {'name': 'atomic_number',   'value': lambda: att_atomic_number(mol_array.element),          'type': 'INT',     'domain': 'POINT'},
        {'name': 'b_factor',        'value': lambda: att_b_factor(mol_array),                       'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'vdw_radii',       'value': lambda: att_vdw_radii(mol_array, world_scale),         'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'chain_id',        'value': lambda: att_chain_id(mol_array),                       'type': 'INT',     'domain': 'POINT'},
        {'name': 'atom_name',       'value': lambda: att_atom_name(mol_array),                      'type': 'INT',     'domain': 'POINT'},
        {'name': 'lipophobicity',   'value': lambda: att_lipophobicity(mol_array),                  'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'charge',          'value': lambda: att_charge(mol_array),                         'type': 'FLOAT',   'domain': 'POINT'},
        {'name': 'is_backbone',     'value': lambda: att_is_backbone(mol_array),                    'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_alpha_carbon', 'value': lambda: att_is_alpha(mol_array),                       'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_solvent',      'value': lambda: att_is_solvent(mol_array),                     'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_nucleic',      'value': lambda: att_is_nucleic(mol_array),                     'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_peptide',      'value': lambda: att_is_peptide(mol_array),                     'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_hetero',       'value': lambda: att_is_hetero(mol_array),                      'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'is_carb',         'value': lambda: att_is_carb(mol_array),                        'type': 'BOOLEAN', 'domain': 'POINT'},
        {'name': 'sec_struct',      'value': lambda: att_sec_struct(mol_array, file, calculate_ss), 'type': 'INT',     'domain': 'POINT'}
    )
    # TODO Make it possible to include / exclude particular attributes.
    # This might boost performance and may or may not be a good idea.
    # Needs testing.

    # Add the attributes to the object
    for att in attributes:
        try:
            obj.add_attribute(mol_object, att['name'], att['value'](), att['type'], att['domain'])
        except:
            warnings.warn(f"Unable to add attribute: {att['name']}")

    if not mol_frames:
        coll_frames = None
    else:
        try:
            b_factors = list(pdb_get_b_factors(file))
        except:
            b_factors = []

        coll_frames = coll.frames(mol_object.name)

        for i, frame in enumerate(mol_frames):
            obj_frame = obj.create_object(
                name=f'{mol_object.name}_frame_{i}',
                collection=coll_frames,
                locations=frame.coord * world_scale - centroid)
            if b_factors:
                try:
                    obj.add_attribute(obj_frame, 'b_factor', b_factors[i])
                except:
                    b_factors.clear()

        # Disable (and thereby hide) the frames collection.
        bpy.context.view_layer.layer_collection.children[collection.name].children[coll_frames.name].exclude = True

    # Add custom properties to the Blender object (no. of chains, biological assemblies etc).
    # NOTE Currently, biological assemblies can be problematic.
    try:
        mol_object['chain_id_unique'] = list(np.unique(mol_array.chain_id))
    except:
        warnings.warn('No chain information detected.')

    return mol_object, coll_frames
