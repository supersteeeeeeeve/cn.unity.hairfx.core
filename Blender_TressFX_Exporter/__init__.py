"""
TressFX Exporter for Blender 4.5+
Export AMD TressFX hair data for Unity HairFX

Author: Maximilian Hoffmann
Version: 1.0
Blender: 4.5+
"""

bl_info = {
    "name": "TressFX Exporter",
    "author": "Maximilian Hoffmann / Community",
    "version": (4, 0, 0),
    "blender": (3, 0, 0),  # Changed to 3.0 for wider compatibility
    "location": "View3D > Sidebar > TressFX",
    "description": "Export AMD TressFX hair data for Unity HairFX",
    "category": "Import-Export",
    "doc_url": "https://github.com/Unity-China/cn.unity.hairfx.core",
}


# Import modules
if "bpy" in locals():
    import importlib
    if "operators" in locals():
        importlib.reload(operators)
    if "ui" in locals():
        importlib.reload(ui)
    if "tfx_formats" in locals():
        importlib.reload(tfx_formats)

import bpy
from . import operators
from . import ui
from . import tfx_formats


# Registration
def register():
    """Register addon"""
    try:
        operators.register()
        ui.register()
        print("TressFX Exporter: Successfully registered")
    except Exception as e:
        print(f"TressFX Exporter: Registration error - {e}")
        import traceback
        traceback.print_exc()


def unregister():
    """Unregister addon"""
    try:
        ui.unregister()
        operators.unregister()
        print("TressFX Exporter: Successfully unregistered")
    except Exception as e:
        print(f"TressFX Exporter: Unregistration error - {e}")


# Allow running as script
if __name__ == "__main__":
    register()
