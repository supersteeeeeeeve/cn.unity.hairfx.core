"""
TressFX Export Operators for Blender
Exports hair curves, skin data, bone data, and collision meshes
"""

import bpy
import bmesh
import mathutils
import ctypes
import random
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper
from .tfx_formats import (
    TressFXTFXFileHeader, TressFXFloat4, TressFXFloat2,
    TressFXSkinFileObject, HairToTriangleMapping,
    WeightJointIndexPair, TRESSFX_MAX_INFLUENTIAL_BONE_COUNT
)


def compute_barycentric_coords(p0, p1, p2, p):
    """Compute barycentric coordinates of point p in triangle (p0, p1, p2)"""
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        # Degenerate triangle
        return [1.0, 0.0, 0.0]

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # Ensure non-negative
    u = max(0.0, u)
    v = max(0.0, v)
    w = max(0.0, w)

    # Normalize
    total = u + v + w
    if total > 0:
        u /= total
        v /= total
        w /= total

    return [u, v, w]


def find_closest_point_on_mesh(point, obj):
    """Find closest point on mesh to given point"""
    # Get world matrix
    mat = obj.matrix_world
    mat_inv = mat.inverted()

    # Transform point to object space
    point_local = mat_inv @ point

    # Find closest point
    result, location, normal, face_index = obj.closest_point_on_mesh(point_local)

    if result:
        # Transform back to world space
        location_world = mat @ location
        return location_world, face_index

    return None, None


def get_hair_curves(context):
    """Get all selected hair curves objects"""
    hair_objects = []
    for obj in context.selected_objects:
        if obj.type == 'CURVES' and hasattr(obj.data, 'curves'):
            hair_objects.append(obj)
    return hair_objects


def get_curve_points(curve_obj, num_samples):
    """Sample points along a curve"""
    curves_data = curve_obj.data
    points = []

    # Blender 4.x uses curves API
    if hasattr(curves_data, 'curves'):
        for curve_idx in range(len(curves_data.curves)):
            curve = curves_data.curves[curve_idx]
            curve_points = []

            # Get points for this curve
            point_start = curve.first_point_index
            point_count = curve.points_length

            # Sample evenly along the curve
            for i in range(num_samples):
                t = i / (num_samples - 1) if num_samples > 1 else 0.0
                point_idx = point_start + int(t * (point_count - 1))
                point_idx = min(point_idx, point_start + point_count - 1)

                # Get position
                pos = curves_data.points[point_idx].position
                world_pos = curve_obj.matrix_world @ mathutils.Vector(pos)
                curve_points.append(world_pos)

            points.append(curve_points)

    return points


class TRESSFX_OT_export_hair(bpy.types.Operator, ExportHelper):
    """Export TressFX Hair Data (.tfx)"""
    bl_idname = "export_scene.tressfx_hair"
    bl_label = "Export TressFX Hair"
    bl_options = {'PRESET'}

    filename_ext = ".tfx"
    filter_glob: StringProperty(default="*.tfx", options={'HIDDEN'})

    num_vertices_per_strand: EnumProperty(
        name="Vertices per Strand",
        items=[
            ('4', '4', '4 vertices per strand'),
            ('8', '8', '8 vertices per strand'),
            ('16', '16', '16 vertices per strand'),
            ('32', '32', '32 vertices per strand'),
            ('64', '64', '64 vertices per strand'),
        ],
        default='16'
    )

    invert_z: BoolProperty(
        name="Invert Z-Axis",
        description="Invert Z-axis of hair strands",
        default=False
    )

    invert_x: BoolProperty(
        name="Invert X-Axis",
        description="Invert X-axis of hair strands",
        default=False
    )

    invert_uv_y: BoolProperty(
        name="Invert UV Y-Axis",
        description="Invert Y-axis of UV coordinates",
        default=False
    )

    randomize_strands: BoolProperty(
        name="Randomize Strands for LOD",
        description="Randomize strand order for better LOD",
        default=True
    )

    base_mesh: StringProperty(
        name="Base Mesh",
        description="Base mesh for UV mapping (leave empty to skip UV export)"
    )

    def execute(self, context):
        # Get all curves from selected hair objects
        hair_objects = get_hair_curves(context)

        if not hair_objects:
            self.report({'ERROR'}, "No hair curves selected!")
            return {'CANCELLED'}

        num_samples = int(self.num_vertices_per_strand)
        all_strands = []

        # Collect all strands from all hair objects
        for hair_obj in hair_objects:
            curves_data = hair_obj.data
            if hasattr(curves_data, 'curves'):
                for curve_idx in range(len(curves_data.curves)):
                    curve = curves_data.curves[curve_idx]
                    strand_points = []

                    point_start = curve.first_point_index
                    point_count = curve.points_length

                    # Sample points along curve
                    for i in range(num_samples):
                        t = i / (num_samples - 1) if num_samples > 1 else 0.0
                        point_idx = point_start + int(t * (point_count - 1))
                        point_idx = min(point_idx, point_start + point_count - 1)

                        pos = curves_data.points[point_idx].position
                        world_pos = hair_obj.matrix_world @ mathutils.Vector(pos)
                        strand_points.append(world_pos)

                    if len(strand_points) > 0:
                        all_strands.append(strand_points)

        if len(all_strands) == 0:
            self.report({'ERROR'}, "No valid hair strands found!")
            return {'CANCELLED'}

        # Randomize if requested
        if self.randomize_strands:
            random.shuffle(all_strands)

        # Get base mesh for UV coordinates
        base_mesh_obj = None
        if self.base_mesh and self.base_mesh in bpy.data.objects:
            base_mesh_obj = bpy.data.objects[self.base_mesh]

        # Export to file
        self.export_tfx_file(all_strands, base_mesh_obj)

        self.report({'INFO'}, f"Exported {len(all_strands)} hair strands to {self.filepath}")
        return {'FINISHED'}

    def export_tfx_file(self, strands, base_mesh_obj):
        """Export TFX binary file"""
        num_strands = len(strands)
        num_vertices_per_strand = len(strands[0]) if strands else 0

        # Create header
        header = TressFXTFXFileHeader()
        header.version = 4.0
        header.numHairStrands = num_strands
        header.numVerticesPerStrand = num_vertices_per_strand
        header.offsetVertexPosition = ctypes.sizeof(TressFXTFXFileHeader)
        header.offsetStrandUV = 0
        header.offsetVertexUV = 0
        header.offsetStrandThickness = 0
        header.offsetVertexColor = 0

        # Calculate UV offset if we have base mesh
        if base_mesh_obj:
            header.offsetStrandUV = (header.offsetVertexPosition +
                                     num_strands * num_vertices_per_strand *
                                     ctypes.sizeof(TressFXFloat4))

        # Write file
        with open(self.filepath, 'wb') as f:
            # Write header
            f.write(header)

            # Write vertex positions
            for strand in strands:
                for i, pos in enumerate(strand):
                    vertex = TressFXFloat4()

                    # Apply coordinate system transformations
                    vertex.x = -pos.x if not self.invert_x else pos.x
                    vertex.y = pos.y
                    vertex.z = -pos.z if self.invert_z else pos.z

                    # Set inverse mass (first two vertices are immovable)
                    vertex.w = 0.0 if i < 2 else 1.0

                    f.write(vertex)

            # Write UV coordinates if base mesh exists
            if base_mesh_obj and base_mesh_obj.type == 'MESH':
                for strand in strands:
                    root_pos = strand[0]

                    # Find UV at root position
                    closest_point, face_idx = find_closest_point_on_mesh(root_pos, base_mesh_obj)

                    uv = TressFXFloat2()
                    uv.x = 0.0
                    uv.y = 0.0

                    if face_idx is not None and hasattr(base_mesh_obj.data, 'uv_layers'):
                        if len(base_mesh_obj.data.uv_layers) > 0:
                            uv_layer = base_mesh_obj.data.uv_layers.active
                            mesh = base_mesh_obj.data
                            face = mesh.polygons[face_idx]

                            # Get average UV of face (simple approach)
                            uv_sum = mathutils.Vector((0, 0))
                            for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                                uv_sum += mathutils.Vector(uv_layer.data[loop_idx].uv)

                            if face.loop_total > 0:
                                uv_avg = uv_sum / face.loop_total
                                uv.x = uv_avg.x
                                uv.y = 1.0 - uv_avg.y if self.invert_uv_y else uv_avg.y

                    f.write(uv)


class TRESSFX_OT_export_skin(bpy.types.Operator, ExportHelper):
    """Export TressFX Skin Data (.tfxskin)"""
    bl_idname = "export_scene.tressfx_skin"
    bl_label = "Export TressFX Skin"
    bl_options = {'PRESET'}

    filename_ext = ".tfxskin"
    filter_glob: StringProperty(default="*.tfxskin", options={'HIDDEN'})

    base_mesh: StringProperty(
        name="Base Mesh",
        description="Base mesh for skin mapping"
    )

    invert_uv_y: BoolProperty(
        name="Invert UV Y-Axis",
        description="Invert Y-axis of UV coordinates",
        default=False
    )

    def execute(self, context):
        # Get hair curves
        hair_objects = get_hair_curves(context)

        if not hair_objects:
            self.report({'ERROR'}, "No hair curves selected!")
            return {'CANCELLED'}

        # Get base mesh
        if not self.base_mesh or self.base_mesh not in bpy.data.objects:
            self.report({'ERROR'}, "Base mesh must be specified!")
            return {'CANCELLED'}

        base_mesh_obj = bpy.data.objects[self.base_mesh]
        if base_mesh_obj.type != 'MESH':
            self.report({'ERROR'}, "Base mesh must be a mesh object!")
            return {'CANCELLED'}

        # Collect root positions
        root_positions = []
        for hair_obj in hair_objects:
            curves_data = hair_obj.data
            if hasattr(curves_data, 'curves'):
                for curve_idx in range(len(curves_data.curves)):
                    curve = curves_data.curves[curve_idx]
                    point_idx = curve.first_point_index
                    pos = curves_data.points[point_idx].position
                    world_pos = hair_obj.matrix_world @ mathutils.Vector(pos)
                    root_positions.append(world_pos)

        if len(root_positions) == 0:
            self.report({'ERROR'}, "No hair roots found!")
            return {'CANCELLED'}

        # Export skin file
        self.export_tfxskin_file(root_positions, base_mesh_obj)

        self.report({'INFO'}, f"Exported skin data for {len(root_positions)} strands")
        return {'FINISHED'}

    def export_tfxskin_file(self, root_positions, base_mesh_obj):
        """Export TFXSkin binary file"""
        mesh = base_mesh_obj.data

        # Ensure mesh has triangles
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

        # Collect mapping data
        mappings = []
        uv_coords = []

        for root_pos in root_positions:
            closest_point, face_idx = find_closest_point_on_mesh(root_pos, base_mesh_obj)

            if face_idx is not None:
                face = mesh.polygons[face_idx]

                # Get triangle vertices
                verts = [mesh.vertices[v].co for v in face.vertices[:3]]
                verts_world = [base_mesh_obj.matrix_world @ v for v in verts]

                # Compute barycentric coordinates
                bary = compute_barycentric_coords(verts_world[0], verts_world[1], verts_world[2], root_pos)

                mappings.append((face_idx, bary))

                # Get UV coordinates
                uv = [0.0, 0.0]
                if hasattr(mesh, 'uv_layers') and len(mesh.uv_layers) > 0:
                    uv_layer = mesh.uv_layers.active
                    uv_sum = mathutils.Vector((0, 0))
                    for loop_idx in range(face.loop_start, face.loop_start + face.loop_total):
                        uv_sum += mathutils.Vector(uv_layer.data[loop_idx].uv)
                    if face.loop_total > 0:
                        uv_avg = uv_sum / face.loop_total
                        uv = [uv_avg.x, 1.0 - uv_avg.y if self.invert_uv_y else uv_avg.y]

                uv_coords.append(uv)
            else:
                mappings.append((0, [1.0, 0.0, 0.0]))
                uv_coords.append([0.0, 0.0])

        # Write file
        header = TressFXSkinFileObject()
        header.version = 1
        header.numHairs = len(mappings)
        header.numTriangles = 0
        header.hairToMeshMap_Offset = ctypes.sizeof(TressFXSkinFileObject)
        header.perStrandUVCoordinate_Offset = (header.hairToMeshMap_Offset +
                                                len(mappings) * ctypes.sizeof(HairToTriangleMapping))

        with open(self.filepath, 'wb') as f:
            f.write(header)

            # Write mappings
            for tri_idx, bary in mappings:
                mapping = HairToTriangleMapping()
                mapping.mesh = 0
                mapping.triangle = tri_idx
                mapping.barycentricCoord_x = bary[0]
                mapping.barycentricCoord_y = bary[1]
                mapping.barycentricCoord_z = bary[2]
                mapping.reserved = 0
                f.write(mapping)

            # Write UV coordinates
            for uv in uv_coords:
                uv_data = TressFXFloat4()  # Reusing Float4 for xyz (uv + 0)
                uv_data.x = uv[0]
                uv_data.y = uv[1]
                uv_data.z = 0.0
                uv_data.w = 0.0
                f.write(uv_data)


class TRESSFX_OT_export_bone(bpy.types.Operator, ExportHelper):
    """Export TressFX Bone Data (.tfxbone)"""
    bl_idname = "export_scene.tressfx_bone"
    bl_label = "Export TressFX Bone"
    bl_options = {'PRESET'}

    filename_ext = ".tfxbone"
    filter_glob: StringProperty(default="*.tfxbone", options={'HIDDEN'})

    base_mesh: StringProperty(
        name="Base Mesh",
        description="Rigged base mesh for bone data"
    )

    def execute(self, context):
        # Get hair curves
        hair_objects = get_hair_curves(context)

        if not hair_objects:
            self.report({'ERROR'}, "No hair curves selected!")
            return {'CANCELLED'}

        # Get base mesh
        if not self.base_mesh or self.base_mesh not in bpy.data.objects:
            self.report({'ERROR'}, "Base mesh must be specified!")
            return {'CANCELLED'}

        base_mesh_obj = bpy.data.objects[self.base_mesh]
        if base_mesh_obj.type != 'MESH':
            self.report({'ERROR'}, "Base mesh must be a mesh object!")
            return {'CANCELLED'}

        # Check for armature modifier
        armature_mod = None
        for mod in base_mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                armature_mod = mod
                break

        if not armature_mod or not armature_mod.object:
            self.report({'ERROR'}, "Base mesh must have an armature modifier!")
            return {'CANCELLED'}

        # Collect root positions
        root_positions = []
        for hair_obj in hair_objects:
            curves_data = hair_obj.data
            if hasattr(curves_data, 'curves'):
                for curve_idx in range(len(curves_data.curves)):
                    curve = curves_data.curves[curve_idx]
                    point_idx = curve.first_point_index
                    pos = curves_data.points[point_idx].position
                    world_pos = hair_obj.matrix_world @ mathutils.Vector(pos)
                    root_positions.append(world_pos)

        if len(root_positions) == 0:
            self.report({'ERROR'}, "No hair roots found!")
            return {'CANCELLED'}

        # Export bone file
        self.export_tfxbone_file(root_positions, base_mesh_obj, armature_mod.object)

        self.report({'INFO'}, f"Exported bone data for {len(root_positions)} strands")
        return {'FINISHED'}

    def export_tfxbone_file(self, root_positions, base_mesh_obj, armature_obj):
        """Export TFXBone binary file"""
        mesh = base_mesh_obj.data

        # Ensure mesh has triangles
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()

        # Get vertex groups (bones)
        vertex_groups = base_mesh_obj.vertex_groups
        bone_names = [vg.name for vg in vertex_groups]

        if len(bone_names) == 0:
            self.report({'ERROR'}, "Base mesh has no vertex groups (bone weights)!")
            return

        # Collect weight data for all vertices
        num_verts = len(mesh.vertices)
        vertex_weights = []

        for v_idx in range(num_verts):
            vert = mesh.vertices[v_idx]
            weights = []

            # Get all weights for this vertex
            for group in vert.groups:
                if group.group < len(bone_names):
                    pair = WeightJointIndexPair()
                    pair.weight = group.weight
                    pair.joint_index = group.group
                    weights.append(pair)

            # Sort by weight (descending)
            weights.sort()

            # Keep only top TRESSFX_MAX_INFLUENTIAL_BONE_COUNT weights
            weights = weights[:TRESSFX_MAX_INFLUENTIAL_BONE_COUNT]

            # Pad with zeros if needed
            while len(weights) < TRESSFX_MAX_INFLUENTIAL_BONE_COUNT:
                pair = WeightJointIndexPair()
                pair.weight = 0.0
                pair.joint_index = 0
                weights.append(pair)

            vertex_weights.append(weights)

        # Collect mapping data (similar to skin export)
        mappings = []

        for root_pos in root_positions:
            closest_point, face_idx = find_closest_point_on_mesh(root_pos, base_mesh_obj)

            if face_idx is not None:
                face = mesh.polygons[face_idx]

                # Get triangle vertices
                vert_indices = list(face.vertices[:3])
                verts = [mesh.vertices[v].co for v in vert_indices]
                verts_world = [base_mesh_obj.matrix_world @ v for v in verts]

                # Compute barycentric coordinates
                bary = compute_barycentric_coords(verts_world[0], verts_world[1], verts_world[2], root_pos)

                # Interpolate weights using barycentric coordinates
                interpolated_weights = self.interpolate_weights(
                    vert_indices, vertex_weights, bary
                )

                mappings.append((face_idx, bary, interpolated_weights))
            else:
                # Default weights
                default_weights = []
                for i in range(TRESSFX_MAX_INFLUENTIAL_BONE_COUNT):
                    pair = WeightJointIndexPair()
                    pair.weight = 1.0 if i == 0 else 0.0
                    pair.joint_index = 0
                    default_weights.append(pair)
                mappings.append((0, [1.0, 0.0, 0.0], default_weights))

        # Write file
        with open(self.filepath, 'wb') as f:
            # Write number of bones
            f.write(ctypes.c_int(len(bone_names)))

            # Write bone names
            for i, bone_name in enumerate(bone_names):
                f.write(ctypes.c_int(i))
                name_bytes = bone_name.encode('utf-8')
                f.write(ctypes.c_int(len(name_bytes) + 1))
                f.write(name_bytes)
                f.write(ctypes.c_byte(0))  # Null terminator

            # Write number of strands
            f.write(ctypes.c_int(len(mappings)))

            # Write bone data for each strand
            for i, (tri_idx, bary, weights) in enumerate(mappings):
                f.write(ctypes.c_int(i))

                # Write 4 bone indices and weights
                for j in range(TRESSFX_MAX_INFLUENTIAL_BONE_COUNT):
                    f.write(ctypes.c_int(weights[j].joint_index))
                    f.write(ctypes.c_float(weights[j].weight))

    def interpolate_weights(self, vert_indices, vertex_weights, bary):
        """Interpolate bone weights using barycentric coordinates"""
        result_weights = {}

        # Accumulate weights from triangle vertices
        for i, vert_idx in enumerate(vert_indices):
            bary_weight = bary[i]
            vert_weight = vertex_weights[vert_idx]

            for weight_pair in vert_weight:
                bone_idx = weight_pair.joint_index
                weight = weight_pair.weight * bary_weight

                if bone_idx in result_weights:
                    result_weights[bone_idx] += weight
                else:
                    result_weights[bone_idx] = weight

        # Convert to sorted list
        final_weights = []
        for bone_idx, weight in result_weights.items():
            pair = WeightJointIndexPair()
            pair.weight = weight
            pair.joint_index = bone_idx
            final_weights.append(pair)

        # Sort by weight (descending)
        final_weights.sort()

        # Keep top TRESSFX_MAX_INFLUENTIAL_BONE_COUNT
        final_weights = final_weights[:TRESSFX_MAX_INFLUENTIAL_BONE_COUNT]

        # Normalize weights
        total_weight = sum(w.weight for w in final_weights)
        if total_weight > 0:
            for w in final_weights:
                w.weight /= total_weight

        # Pad with zeros if needed
        while len(final_weights) < TRESSFX_MAX_INFLUENTIAL_BONE_COUNT:
            pair = WeightJointIndexPair()
            pair.weight = 0.0
            pair.joint_index = 0
            final_weights.append(pair)

        return final_weights


class TRESSFX_OT_export_collision(bpy.types.Operator, ExportHelper):
    """Export TressFX Collision Mesh (.tfxmesh)"""
    bl_idname = "export_scene.tressfx_collision"
    bl_label = "Export TressFX Collision Mesh"
    bl_options = {'PRESET'}

    filename_ext = ".tfxmesh"
    filter_glob: StringProperty(default="*.tfxmesh", options={'HIDDEN'})

    def execute(self, context):
        if not context.active_object or context.active_object.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object!")
            return {'CANCELLED'}

        mesh_obj = context.active_object
        self.export_collision_mesh(mesh_obj)

        self.report({'INFO'}, f"Exported collision mesh: {mesh_obj.name}")
        return {'FINISHED'}

    def export_collision_mesh(self, mesh_obj):
        """Export collision mesh as text file"""
        mesh = mesh_obj.data

        # Ensure mesh is triangulated
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)

        with open(self.filepath, 'w') as f:
            f.write("# TressFX collision mesh exported from Blender\n")
            f.write("numOfBones 0\n")

            # Write vertices
            f.write(f"numOfVertices {len(bm.verts)}\n")
            f.write("# vertex index, position xyz, normal xyz, bone indices (4), weights (4)\n")

            for i, vert in enumerate(bm.verts):
                pos = mesh_obj.matrix_world @ vert.co
                normal = mesh_obj.matrix_world.to_3x3() @ vert.normal
                f.write(f"{i} {pos.x} {pos.y} {pos.z} {normal.x} {normal.y} {normal.z} ")
                f.write("0 0 0 0 1.0 0.0 0.0 0.0\n")

            # Write triangles
            f.write(f"numOfTriangles {len(bm.faces)}\n")
            f.write("# triangle index, vertex indices (3)\n")

            for i, face in enumerate(bm.faces):
                verts = [v.index for v in face.verts]
                f.write(f"{i} {verts[0]} {verts[1]} {verts[2]}\n")

        bm.free()


# Registration
classes = (
    TRESSFX_OT_export_hair,
    TRESSFX_OT_export_skin,
    TRESSFX_OT_export_bone,
    TRESSFX_OT_export_collision,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
