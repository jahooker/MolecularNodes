import bpy
from bpy.types import Mesh, Object, Attribute, Collection
import numpy as np
from typing import Optional


def create_object(name: str, collection: Collection, locations, bonds: Optional[list]) -> Object:
    """
    Creates a mesh with the given name in the given collection, from the supplied
    values for the locations of vertices, and if supplied, bonds as edges.
    """
    mol_mesh: Mesh = bpy.data.meshes.new(name)
    mol_mesh.from_pydata(locations, bonds or [], faces=[])
    mol_object: Object = bpy.data.objects.new(name, mol_mesh)
    collection.objects.link(mol_object)
    return mol_object


def add_attribute(obj: Object, name: str, data, data_type: str = 'FLOAT', domain: str = 'POINT'):
    attribute: Attribute = obj.data.attributes.new(name, data_type, domain)
    attribute.data.foreach_set('value', data)


def get_attribute(obj: Object, attr_name: str = 'position') -> Optional[np.ndarray]:
    """ Retrieve Attribute from Object as numpy.ndarray
    """
    att: Attribute = obj.to_mesh().attributes[attr_name]
    try:
        dtype = {'INT': int, 'FLOAT': float, 'BOOLEAN': bool}[att.data_type]
        return np.array([x.value for x in att.data.values()], dtype=dtype)
    except KeyError:
        if att.data_type == 'FLOAT_VECTOR':
            return np.array([x.vector for x in att.data.values()])


class AttributeGetter:
    ''' A function with associated `name`, `data_type`, and `domain`.
    '''

    def __init__(self, name: str, get, data_type: str, domain: str = 'POINT'):
        self.name = name
        self.get = get
        self.data_type = data_type
        self.domain = domain

    def __str__(self) -> str:
        return self.name

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    @classmethod
    def from_function(cls, name, data_type, domain):
        return lambda f: AttributeGetter(name, f, data_type, domain)  # Use this as decoration
