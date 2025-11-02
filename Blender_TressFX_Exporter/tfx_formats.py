"""
TressFX Binary File Format Structures
"""

import ctypes


# Maximum number of influential bones per vertex
TRESSFX_MAX_INFLUENTIAL_BONE_COUNT = 4


class TressFXTFXFileHeader(ctypes.Structure):
    """Header for .tfx hair file"""
    _fields_ = [
        ('version', ctypes.c_float),
        ('numHairStrands', ctypes.c_uint),
        ('numVerticesPerStrand', ctypes.c_uint),
        ('offsetVertexPosition', ctypes.c_uint),
        ('offsetStrandUV', ctypes.c_uint),
        ('offsetVertexUV', ctypes.c_uint),
        ('offsetStrandThickness', ctypes.c_uint),
        ('offsetVertexColor', ctypes.c_uint),
        ('reserved', ctypes.c_uint * 32)
    ]


class TressFXFloat4(ctypes.Structure):
    """4D vector for vertex positions"""
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('w', ctypes.c_float)  # w is inverse mass (0 = immovable)
    ]


class TressFXFloat2(ctypes.Structure):
    """2D vector for UV coordinates"""
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float)
    ]


class TressFXSkinFileObject(ctypes.Structure):
    """Header for .tfxskin file"""
    _fields_ = [
        ('version', ctypes.c_uint),
        ('numHairs', ctypes.c_uint),
        ('numTriangles', ctypes.c_uint),
        ('reserved1', ctypes.c_uint * 31),
        ('hairToMeshMap_Offset', ctypes.c_uint),
        ('perStrandUVCoordinate_Offset', ctypes.c_uint),
        ('reserved2', ctypes.c_uint * 31)
    ]


class HairToTriangleMapping(ctypes.Structure):
    """Mapping of hair strand root to triangle with barycentric coordinates"""
    _fields_ = [
        ('mesh', ctypes.c_uint),
        ('triangle', ctypes.c_uint),
        ('barycentricCoord_x', ctypes.c_float),
        ('barycentricCoord_y', ctypes.c_float),
        ('barycentricCoord_z', ctypes.c_float),
        ('reserved', ctypes.c_uint)
    ]


class WeightJointIndexPair:
    """Helper class for bone weight sorting"""
    def __init__(self):
        self.weight = 0.0
        self.joint_index = -1

    def __lt__(self, other):
        """Sort by weight (descending)"""
        return self.weight > other.weight
