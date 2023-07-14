import bpy
from bpy.types import Collection
from typing import Optional


def mn() -> Collection:
    """Return the MolecularNodes Collection

    The collection called 'MolecularNodes' inside the Blender scene is returned.
    If the collection does not yet exist, it is created.
    """
    if (coll := bpy.data.collections.get('MolecularNodes')) is None:
        coll = bpy.data.collections.new('MolecularNodes')
        bpy.context.scene.collection.children.link(coll)
    return coll


def frames(name: str = "", parent: Optional[Collection] = None, format=lambda s: f"{s}_frames"):
    """Create a Collection for Frames of a Trajectory

    Args:
        name (str): Name of the collection for the frames. Defaults to "".
        parent (Collection, optional): A blender collection which will become the parent collection.
            Defaults to the MolecularNodes collection.
    """
    coll_frames = bpy.data.collections.new(format(name))
    (mn() if parent is None else parent).children.link(coll_frames)
    return coll_frames
