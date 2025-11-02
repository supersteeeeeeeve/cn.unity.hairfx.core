"""
TressFX UI Panel
"""

import bpy
from bpy.types import Panel


class TRESSFX_PT_main_panel(Panel):
    """TressFX Exporter Main Panel"""
    bl_label = "TressFX Exporter"
    bl_idname = "TRESSFX_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TressFX'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Header
        layout.label(text="AMD TressFX Hair Exporter", icon='OUTLINER_OB_CURVES')
        layout.separator()

        # Export Hair Section
        box = layout.box()
        box.label(text="Hair/Fur Export", icon='CURVES_DATA')

        col = box.column(align=True)
        col.label(text="Select hair curves and export:")
        col.operator("export_scene.tressfx_hair", text="Export Hair Data (.tfx)", icon='EXPORT')

        layout.separator()

        # Export Skin Section
        box = layout.box()
        box.label(text="Skin Data Export", icon='SURFACE_DATA')

        col = box.column(align=True)
        col.label(text="Export hair to mesh mapping:")
        col.operator("export_scene.tressfx_skin", text="Export Skin Data (.tfxskin)", icon='EXPORT')

        layout.separator()

        # Export Bone Section
        box = layout.box()
        box.label(text="Bone Data Export", icon='ARMATURE_DATA')

        col = box.column(align=True)
        col.label(text="Export bone weights:")
        col.operator("export_scene.tressfx_bone", text="Export Bone Data (.tfxbone)", icon='EXPORT')

        layout.separator()

        # Export Collision Section
        box = layout.box()
        box.label(text="Collision Mesh Export", icon='MESH_DATA')

        col = box.column(align=True)
        col.label(text="Export collision mesh:")
        col.operator("export_scene.tressfx_collision", text="Export Collision (.tfxmesh)", icon='EXPORT')

        layout.separator()

        # Info
        layout.label(text="TressFX Exporter v4.0")
        layout.label(text="Compatible with Unity HairFX")


# Registration
classes = (
    TRESSFX_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
