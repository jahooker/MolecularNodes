import bpy
from . import nodes
from . import pkg
from . import load
from . import md
from . import assembly
from . import density
from . import star
from . import esmfold
from . import density


class BlenderContext:

    def __init__(self, inner: str):
        self.outer = None
        self.inner = inner

    def __enter__(self):
        self.outer = bpy.context.area.type
        bpy.context.area.type = self.inner

    def __exit__(self, exc_type, exc_value, exc_traceback):
        bpy.context.area.type = self.outer
        self.outer = None


class Pollable:

    @classmethod
    def poll(cls, context):
        return True


class Menu(bpy.types.Menu, Pollable):

    @classmethod
    @property
    def bl_idname(cls) -> str:
        return cls.__name__

    bl_label: str = ''

    def interface(
        self, label: str, node_name: str,
        node_description: str = 'Add custom MolecularNodes node group.'
    ):
        op = self.layout.operator(MOL_OT_Add_Custom_Node_Group.bl_idname,
                                  text=label, emboss=True, depress=False)
        op.node_name = node_name
        op.node_description = node_description
        return op


class Panel(bpy.types.Panel, Pollable):

    @classmethod
    @property
    def bl_idname(cls) -> str:
        return cls.__name__

    bl_label: str = ''


class Operator(bpy.types.Operator, Pollable):

    @classmethod
    @property
    def bl_idname(cls) -> str:
        return 'mol.' + cls.__name__.removeprefix('MOL_OT_').lower()

    bl_label:       str  = ''
    bl_description: str  = ''
    bl_options: set[str] = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return self.execute(context)

    @staticmethod
    def add_node(node_name):
        with BlenderContext('NODE_EDITOR'):
            # Add a node to the current node tree
            bpy.ops.node.add_node(
                'INVOKE_DEFAULT', type='GeometryNodeGroup', use_transform=True)
                # use_transform=True ensures that the new node appears where the user's mouse is
                # and is currently being moved so the user can place it where they wish
        bpy.context.active_node.node_tree = bpy.data.node_groups[node_name]
        bpy.context.active_node.width = 200.0
        # If added node has a 'Material' input, set it to the default MN material
        if (input_mat := bpy.context.active_node.inputs.get('Material')) is not None:
            input_mat.default_value = nodes.mol_base_material()


class MOL_OT_Import_Protein_RCSB(Operator):
    ''' Operator that imports a structure from the PDB.
    '''
    bl_label = "import_protein_fetch_pdb"
    bl_description = "Download and open a structure from the Protein Data Bank"

    def execute(self, context):
        pdb_code = bpy.context.scene.mol_pdb_code
        mol_object = load.molecule_rcsb(
            pdb_code=pdb_code,
            center_molecule=bpy.context.scene.mol_import_center,
            del_solvent=bpy.context.scene.mol_import_del_solvent,
            include_bonds=bpy.context.scene.mol_import_include_bonds,
            starting_style=bpy.context.scene.mol_import_default_style)
        bpy.context.view_layer.objects.active = mol_object
        self.report({'INFO'}, message=f"Imported '{pdb_code}' as {mol_object.name}")
        return {"FINISHED"}


class MOL_OT_Import_Protein_Local(Operator):
    ''' Operator that imports a structure from a local file.
    '''
    bl_label = "import_protein_local"
    bl_description = "Open a local structure file"

    def execute(self, context):
        file_path = bpy.context.scene.mol_import_local_path
        mol_object = load.molecule_local(
            file_path=file_path,
            mol_name=bpy.context.scene.mol_import_local_name,
            include_bonds=bpy.context.scene.mol_import_include_bonds,
            center_molecule=bpy.context.scene.mol_import_center,
            del_solvent=bpy.context.scene.mol_import_del_solvent,
            default_style=bpy.context.scene.mol_import_default_style,
            setup_nodes=True)
        bpy.context.view_layer.objects.active = mol_object
        self.report({'INFO'}, message=f"Imported '{file_path}' as {mol_object.name}")
        return {"FINISHED"}


class MOL_OT_Import_Method_Selection(Operator):

    bl_label = "import_method"
    bl_description = "Change Structure Import Method"
    mol_interface_value: bpy.props.IntProperty(
        name='interface_value', description='', default=0, subtype='NONE')

    def execute(self, context):
        bpy.context.scene.mol_import_panel_selection = self.mol_interface_value
        return {"FINISHED"}


class MOL_OT_Default_Style(Operator):

    bl_label = "Change the default style."
    bl_description = "Change the default style of molecules on import."
    panel_display: bpy.props.IntProperty(name='panel_display', default=0)

    def execute(self, context):
        bpy.context.scene.mol_import_default_style = self.panel_display
        return {"FINISHED"}


class MOL_MT_Default_Style(Menu):

    def draw(self, context):
        layout = self.layout.column_flow(columns=1)
        for i, label in enumerate(['Atoms', 'Cartoon', 'Ribbon', 'Ball and Stick']):
            depress = i == bpy.context.scene.mol_import_default_style
            op = layout.operator(MOL_OT_Default_Style.bl_idname,
                                 text=label, emboss=True, depress=depress)
            op.panel_display = i


class MOL_PT_panel(Panel):

    bl_label = 'Molecular Nodes'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'scene'
    bl_order = 0
    bl_options = {'HEADER_LAYOUT_EXPAND'}
    bl_ui_units_x = 0

    def draw_header(self, context):
        pass

    def draw(self, context):
        self.ui(self.layout, bpy.context.scene)

    @classmethod
    def ui(cls, layout, scene):
        layout.label(text="Import Options", icon="MODIFIER")
        grid = layout.box().grid_flow(columns=2)

        grid.prop(bpy.context.scene, 'mol_import_center',
                text='Centre Structure', icon_value=0, emboss=True)
        grid.prop(bpy.context.scene, 'mol_import_del_solvent',
                text='Delete Solvent', icon_value=0, emboss=True)
        grid.prop(bpy.context.scene, 'mol_import_include_bonds',
                text='Import Bonds', icon_value=0, emboss=True)
        i: int = bpy.context.scene.mol_import_default_style
        grid.menu('MOL_MT_Default_Style',
                text=['Atoms', 'Cartoon', 'Ribbon', 'Ball and Stick'][i])
        panel = layout
        # row = panel.row(heading='', align=True)
        row = panel.grid_flow(row_major=True, columns=3, align=True)
        row.alignment = 'EXPAND'
        row.enabled = True
        row.alert = False

        cls.change_import_interface(row, 'PDB',           0, "URL")
        cls.change_import_interface(row, 'ESMFold',       1, "URL")
        cls.change_import_interface(row, 'Local File',    2, 108)
        cls.change_import_interface(row, 'MD Trajectory', 3, 487)
        cls.change_import_interface(row, 'EM Map',        4, "LIGHTPROBE_CUBEMAP")
        cls.change_import_interface(row, 'Star File',     5, 487)

        panel_selection: int = bpy.context.scene.mol_import_panel_selection
        box = panel.column().box()

        def nag(box, package_name: str):
            box.enabled = False
            box.alert = True
            box.label(text=f"Please install {package_name} in the addon preferences.")

        match panel_selection:
            case 0:
                if not pkg.is_current('biotite'):
                    nag(box, 'biotite')
                cls.rscb(box)
            case 1:
                if not pkg.is_current('biotite'):
                    nag(box, 'biotite')
                esmfold.panel(box)
            case 2:
                if not pkg.is_current('biotite'):
                    nag(box, 'biotite')
                cls.local(box)
            case 3:
                if not pkg.is_current('MDAnalysis'):
                    nag(box, 'MDAnalysis')
                md.panel(box, scene)
            case 4:
                if not pkg.is_current('mrcfile'):
                    nag(box, 'mrcfile')
                density.panel(box, scene)
            case 5:
                for package_name in ['starfile', 'eulerangles']:
                    if not pkg.is_current(package_name):
                        nag(box, package_name)
                star.panel(box, scene)

    @staticmethod
    def change_import_interface(
        layout, label: str, interface_value: int, icon: str | int
    ):
        depress = interface_value == bpy.context.scene.mol_import_panel_selection
        kwargs = {'text': label, 'emboss': True, 'depress': depress}
        match icon:
            case str(_):
                kwargs['icon'] = icon
            case int(_):
                kwargs['icon_value'] = icon
        op = layout.operator(MOL_OT_Import_Method_Selection.bl_idname, **kwargs)
        op.mol_interface_value = interface_value
        return op

    @staticmethod
    def rcsb(layout):
        col_main = layout.column(heading='', align=False)
        col_main.alert = False
        col_main.enabled = True
        col_main.active = True
        col_main.use_property_split = False
        col_main.use_property_decorate = False
        col_main.scale_x = 1.0
        col_main.scale_y = 1.0
        col_main.alignment = 'Expand'.upper()
        col_main.label(text="Download from PDB")
        row_import = col_main.row()
        row_import.prop(bpy.context.scene, 'mol_pdb_code', text='PDB ID')
        row_import.operator(MOL_OT_Import_Protein_RCSB.bl_idname, text='Download', icon='IMPORT')

    @staticmethod
    def local(layout):
        col_main = layout.column(heading='', align=False)
        col_main.alert = False
        col_main.enabled = True
        col_main.active = True
        col_main.label(text="Open Local File")
        row_name = col_main.row(align=False)
        row_name.prop(bpy.context.scene, 'mol_import_local_name',
                    text="Name", icon_value=0, emboss=True)
        row_name.operator(MOL_OT_Import_Protein_Local.bl_idname, 
                        text="Load", icon='FILE_TICK', emboss=True)
        row_import = col_main.row()
        row_import.prop(
            bpy.context.scene, 'mol_import_local_path',
            text="File path", icon_value=0, emboss=True)


class MOL_OT_Add_Custom_Node_Group(Operator):

    bl_label = "Add Custom Node Group"
    # bl_description = "Add Molecular Nodes custom node group."
    node_name: bpy.props.StringProperty(
        name="node_name", description="",
        default="", subtype="NONE", maxlen=0)
    node_description: bpy.props.StringProperty(
        name="node_description", description="",
        default="Add MolecularNodes custom node group.", subtype="NONE")

    @staticmethod
    def description(context, properties):
        return properties.node_description

    def execute(self, context):
        try:
            nodes.mol_append_node(self.node_name)
            self.add_node(self.node_name)
        except RuntimeError:
            self.report({'ERROR'},
                        message='Failed to add node. Ensure you are not in edit mode.')
        return {"FINISHED"}


class MOL_OT_Style_Surface_Custom(Operator):

    bl_description = "Create a split surface representation.\n" \
        "Generates a surface based on atomic VDW radii. " \
        "Each chain gets its own surface representation."

    def execute(self, context):
        try:
            node_surface = nodes.create_custom_surface(
                name=f'MOL_style_surface_{context.active_object.name}_split',
                n_chains=len(context.active_object['chain_id_unique']))
        except:
            node_surface = nodes.mol_append_node('MOL_style_surface_single')
            self.report({'WARNING'}, message='Unable to detect number of chains.')
        self.add_node(node_surface.name)
        return {"FINISHED"}


class MOL_OT_Assembly_Bio(Operator):

    bl_label = "Build"
    bl_description = "**Structures Downloaded From PDB Only**\n" \
        "Adds node to build biological assembly based on symmetry operations from the structure file. " \
        "Currently this is only supported for structures downloaded from the PDB."

    def execute(self, context):
        try:
            transformations = assembly.get_transformations_mmtf(context.active_object['bio_transform_dict'])
            node_bio_assembly = assembly.create_biological_assembly_node(
                name=context.active_object.name, transforms=next(transformations))
            # Currently, we only get transformations for the first biological assembly.
            # To extract the rest would require that we adjust node creation.
        except:
            self.report({'WARNING'}, message='Unable to detect biological assembly information.')
            return {"FINISHED"}
        if node_bio_assembly is not None:
            self.add_node(node_bio_assembly.name)
        return {"FINISHED"}


class MOL_OT_Color_Chain(Operator):

    bl_description = "Create a custom node to color each chain of a structure individually.\n" \
                     "Requires that chain information be available."

    def execute(self, context):
        try:
            node_color_chain = nodes.chain_color(
                node_name=f"MOL_color_chains_{context.active_object.name}",
                input_list=context.active_object['chain_id_unique'])
            self.add_node(node_color_chain.name)
        except:
            self.report({'WARNING'}, message='Unable to detect chain information.')
        return {"FINISHED"}


class MOL_OT_Chain_Selection_Custom(Operator):

    bl_label = "Chain Selection"
    bl_description = "Create a selection based on the chains.\n" \
        "This node is built on a per-molecule basis, " \
        "taking into account the chain IDs that were detected. " \
        "If no chain information is available, this node will not work."

    def execute(self, context):
        node_chains = nodes.chain_selection(
            node_name=f"MOL_sel_{bpy.context.view_layer.objects.active.name}_chains",
            chain_names=bpy.context.view_layer.objects.active["chain_id_unique"],
            attribute="chain_id",
            format=lambda s: f"Chain {s}")
        self.add_node(node_chains.name)
        return {"FINISHED"}


class MOL_OT_Residues_Selection_Custom(Operator):

    bl_label = "Multiple Residue Selection"
    bl_description = "Create a selection based on the provided residue strings.\n" \
        "This node is built on a per-molecule basis, " \
        "taking into account the residues that were input."

    input_resid_string: bpy.props.StringProperty(
        name="Select residue IDs: ",
        description="Enter a string value.",
        default="19,94,1-16")

    def execute(self, context):
        node_residues = nodes.resid_multiple_selection(
            node_name='MOL_sel_residues',
            input_resid_string=self.input_resid_string)
        self.add_node(node_residues.name)
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class MOL_OT_Ligand_Selection_Custom(Operator):

    bl_label = "Ligand Selection"
    bl_description = "Create a selection based on the ligands.\n" \
        "This node is built on a per-molecule basis, " \
        "taking into account the chain IDs that were detected. " \
        "If no chain information is available, this node will not work."

    def execute(self, context):
        active_object = bpy.context.view_layer.objects.active
        node_chains = nodes.chain_selection(
            node_name=f"MOL_sel_{active_object.name}_ligands",
            chain_names=active_object['ligands'],
            start=100, attribute='res_name')
        self.add_node(node_chains.name)
        return {"FINISHED"}


class MOL_MT_Add_Node_Menu_Properties(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        # currently nothing for this menu in the panel


class MOL_MT_Add_Node_Menu_Color(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Set Color', 'MOL_color_set',
            "Sets a new color for the selected atoms")
        self.interface('Set Color Common', 'MOL_color_set_common',
            "Choose a color for the most common elements in PDB structures")
        self.layout.separator()
        self.interface('Goodsell Colors', 'MOL_color_goodsell',
            "Adjusts the given colors to copy the 'Goodsell Style'.\n "
            "Darkens the non-carbon atoms and keeps the carbon atoms the same color. "
            "Highlights differences without being too visually busy.")
        self.layout.separator()
        self.interface('Color by SS', 'MOL_color_sec_struct',
            "Specify colors based on the secondary structure.")
        self.interface('Color by Atomic Number', 'MOL_color_atomic_number',
            "Creates a color based on atomic_number field")
        self.interface('Color by Element', 'MOL_color_element',
            "Choose a color for each of the first 20 elements.")
        self.color_chains('Color by Chains')

    def color_chains(self, label):
        return self.layout.operator(MOL_OT_Color_Chain.bl_idname,
                                    text=label, emboss=True, depress=True)


class MOL_MT_Add_Node_Menu_Bonds(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Find Bonds', 'MOL_bonds_find',
            "Finds bonds between atoms based on distance.\n"
            "Based on the vdw_radii for each point, "
            "finds other points within a certain radius to create a bond to. "
            "Does not preserve the index for the points. "
            "Does not detect bond type.")
        self.interface('Break Bonds', 'MOL_bonds_break',
            "Will delete a bond between atoms that already exists "
            "based on a distance cutoff")
        self.interface('Find Bonded Atoms', 'MOL_bonds_find_bonded',
            "Based on an initial selection and some integer n, "
            "finds atoms at most n bonds away.")


class MOL_MT_Add_Node_Menu_Styling(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Atoms Cycles', 'MOL_style_atoms_cycles',
            'A sphere atom representation, visible ONLY in Cycles. \
            Based on point-cloud rendering.')
        self.interface('Atoms EEVEE', 'MOL_style_atoms_eevee',
            'A sphere atom representation, visible in EEVEE and Cycles. \
            Based on mesh instancing, which slows down viewport performance.')
        self.interface('Cartoon', 'MOL_style_cartoon',
            'Create a cartoon representation, \
            highlighting secondary structure through arrows and ribbons.')
        self.interface('Ribbon Protein', 'MOL_style_ribbon_protein',
            'Create a ribbon mesh based off of the alpha-carbons of the structure')
        self.interface('Ribbon Nucleic', 'MOL_style_ribbon_nucleic',
            'Create a ribbon mesh and instanced cylinders for nucleic acids.')
        self.interface('Surface', 'MOL_style_surface_single',
            "Create a single joined surface representation. \
            Generates an isosurface based on atomic vdw_radii. \
            All chains are part of the same surface. \
            Use Surface Split Chains to have a single surface per chain.")
        self.surface_custom('Surface Split Chains')
        self.interface('Ball and Stick', 'MOL_style_ball_and_stick',
            "A style node to create ball and stick representation. \
            Icospheres are instanced on atoms and cylinders for bonds. \
            Bonds can be detected if they are not present in the structure.")

    def surface_custom(self, label):
        return self.layout.operator(MOL_OT_Style_Surface_Custom.bl_idname,
                                    text=label, emboss=True, depress=True)


class MOL_MT_Add_Node_Menu_Selections(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Select Atoms', 'MOL_sel_atoms',
            "Separate atoms based on a selection field.\n "
            "Takes atoms and splits them into the selected atoms the inverted atoms, based on a selection field.")
        self.interface('Separate Polymers', 'MOL_sel_sep_polymers',
            "Separate the Geometry into the different polymers.\n "
            "Outputs for protein, nucleic & sugars.")
        self.layout.separator()
        self.menu_chain_selection_custom(bpy.context.view_layer.objects.active)
        self.menu_ligand_selection_custom(bpy.context.view_layer.objects.active)
        self.layout.separator()
        self.interface('Backbone', 'MOL_sel_backbone',
            "Select atoms it they are part of the side chains or backbone.")
        self.interface('Atom Properties', 'MOL_sel_atom_propeties',
            "Create a selection based on the properties of the atom.\n "
            "Fields for is_alpha_carbon, is_backbone, is_peptide, is_nucleic, is_solvent and is_carb.")
        self.interface('Atomic Number', 'MOL_sel_atomic_number',
            "Create a selection if input value equal to the atomic_number field.")
        self.interface('Element Name', 'MOL_sel_element_name',
            "Create a selection of particular elements by name. "
            "Only first 20 elements supported.")
        self.layout.separator()
        self.interface('Distance', 'MOL_sel_distance',
            "Create a selection based on the distance to a selected object.\n "
            "The cutoff is scaled based on the objects scale and the 'Scale Cutoff' value.")
        self.interface('Slice', 'MOL_sel_slice',
            "Create a selection that is a slice along one of the XYZ axes, "
            "based on the position of an object.")
        self.layout.separator()
        self.menu_residues_selection_custom()
        self.interface('Res ID Single', 'MOL_sel_res_id',
            "Create a selection if res_id matches input field.")
        self.interface('Res ID Range', 'MOL_sel_res_id_range',
            "Create a selection if the res_id is within the given thresholds.")
        self.interface('Res Name Peptide', 'MOL_sel_res_name',
            "Create a selection of particular amino acids by name.")
        self.interface('Res Name Nucleic', 'MOL_sel_res_name_nucleic',
            "Create a selection of particular nucleic acids by name.")
        self.interface('Res Whole', 'MOL_sel_res_whole',
            "Expand the selection to every atom in a residue, "
            "if any of those atoms are in the initial selection.")
        self.interface('Res Atoms', 'MOL_sel_res_atoms',
            "Create a selection based on the atoms of a residue.\n "
            "Selections for CA, backbone atoms (N, C, O), sidechain and backbone.")

    def menu_chain_selection_custom(self, obj):
        return self.layout.operator(MOL_OT_Chain_Selection_Custom.bl_idname,
                                    text=f'Chain {obj.name}', emboss=True, depress=True)

    def menu_ligand_selection_custom(self, obj):
        return self.layout.operator(MOL_OT_Ligand_Selection_Custom.bl_idname,
                                    text=f'Ligands {obj.name}', emboss=True, depress=True)

    def menu_residues_selection_custom(self):
        return self.layout.operator(MOL_OT_Residues_Selection_Custom.bl_idname,
                                    text='Res ID', emboss=True, depress=True)


class MOL_MT_Add_Node_Menu_Assembly(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.layout.operator(MOL_OT_Assembly_Bio.bl_idname,
                             text="Biological Assembly", emboss=True, depress=True)
        self.interface('Center Assembly', 'MOL_assembly_center',
            "Center the structure on the world origin based on bounding box.")


class MOL_MT_Add_Node_Menu_Membranes(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Setup Atomic Properties', 'MOL_prop_setup')


class MOL_MT_Add_Node_Menu_DNA(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Double Helix', 'MOL_dna_double_helix',
            "Create a DNA double helix from an input curve.\n"
            "Takes an input curve and instances for the bases. "
            "Returns instances of the bases in a double helix formation.")
        self.interface('Bases', 'MOL_dna_bases',
            "Provide the DNA bases as instances to be styled and passed onto the Double Helix node.")
        self.layout.separator()
        self.interface('Style Atoms Cycles', 'MOL_dna_style_atoms_cycles',
            "Style the DNA bases with spheres only visible in Cycles.")
        self.interface('Style Atoms EEVEE', 'MOL_dna_style_atoms_eevee',
            "Style the DNA bases with spheres visible in Cycles and EEVEE.")
        self.interface('Style Surface', 'MOL_dna_style_surface',
            "Style the DNA bases with surface representation.")
        self.interface('Style Ball and Stick', 'MOL_dna_style_ball_and_stick',
            "Style the DNA bases with ball and stick representation.")


class MOL_MT_Add_Node_Menu_Animation(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Animate Frames', 'MOL_animate_frames',
            "Interpolate between frames of a trajectory. "
            "Given a collection of frames for a trajectory, "
            "this node interpolates between them from start to finish "
            "based on the Animate field taking a value from 0 to 1. "
            "The atom positions are then moved based on this field.")
        self.interface('Animate Field', 'MOL_animate_field')
        self.interface('Animate Value', 'MOL_animate_value',
            "Animate between given start and end values, "
            "based on the input start and end frame of the timeline. "
            "Clamped will limit the output to the 'To Min' and 'To Max', "
            "while unclamped will continue to interpolate past these values."
            "'Smoother Step' will ease in and out of these values, "
            "with default being linear interpolation.")
        self.layout.separator()
        self.interface('Res Wiggle', "MOL_animate_res_wiggle",
            "Wiggle the side chains of amino acids based on B-factor, "
            "adding movement to a structure.")
        self.interface('Res to Curve', "MOL_animate_res_to_curve",
            "Map atoms along a curve, as a single polypeptide.")
        self.layout.separator()
        self.interface('Noise Position', 'MOL_noise_position',
            "Generate 3D noise field based on the position attribute.")
        self.interface('Noise Field', 'MOL_noise_field',
            "Generate a 3D noise field based on the given field.")
        self.interface('Noise Repeat', 'MOL_noise_repeat',
            "Generate a repeating 3D noise field, based on the given field.")


class MOL_MT_Add_Node_Menu_Utilities(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Boolean Chain', 'MOL_utils_bool_chain')
        self.interface('Rotation Matrix', 'MOL_utils_rotation_matrix')
        self.interface('Curve Resample', 'MOL_utils_curve_resample')
        self.interface('Determine Secondary Structure', 'MOL_utils_dssp')


class MOL_MT_Add_Node_Menu_Density(Menu):

    def draw(self, context):
        self.layout.operator_context = "INVOKE_DEFAULT"
        self.interface('Style Surface', 'MOL_style_density_surface')
        self.interface('Style Wire', 'MOL_style_density_wire')
        self.interface('Sample Nearest Attribute', 'MOL_utils_sample_searest')


class MOL_MT_Add_Node_Menu(Menu):

    bl_label = "Menu for Adding Nodes in GN Tree"

    def draw(self, context):
        layout = self.layout.column_flow(columns=1)
        layout.operator_context = "INVOKE_DEFAULT"
        layout.menu(MOL_MT_Add_Node_Menu_Styling.bl_idname,
                    text='Style', icon_value=77)
        layout.menu(MOL_MT_Add_Node_Menu_Color.bl_idname,
                    text='Color', icon='COLORSET_07_VEC')
        layout.menu(MOL_MT_Add_Node_Menu_Density.bl_idname,
                    text='Density', icon='LIGHTPROBE_CUBEMAP')
        layout.menu(MOL_MT_Add_Node_Menu_Bonds.bl_idname,
                    text='Bonds', icon='FIXED_SIZE')
        layout.menu(MOL_MT_Add_Node_Menu_Selections.bl_idname,
                    text='Selection', icon_value=256)
        layout.menu(MOL_MT_Add_Node_Menu_Animation.bl_idname,
                    text='Animation', icon_value=409)
        layout.menu(MOL_MT_Add_Node_Menu_Assembly.bl_idname,
                    text='Assemblies', icon = 'GROUP_VERTEX')
        layout.menu(MOL_MT_Add_Node_Menu_DNA.bl_idname,
                    text='DNA', icon='GP_SELECT_BETWEEN_STROKES')
        layout.menu(MOL_MT_Add_Node_Menu_Utilities.bl_idname,
                    text='Utilities', icon_value=92)

    def add_node_menu(self, context):
        if bpy.context.area.spaces[0].tree_type == 'GeometryNodeTree':
            self.layout.menu(MOL_MT_Add_Node_Menu.bl_idname, text='Molecular Nodes', icon_value=88)
