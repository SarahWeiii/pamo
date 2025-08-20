import os
import bpy
import numpy
import ctypes


bl_info = {
    "name": "GPU Decimate Plugin",
    "description": "Decimate an manifold, intersect-free triangle mesh on GPU",
    "author": "UCSD Hao Su Lab",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "Modifiers",
    "support": "COMMUNITY",
    "category": "Mesh",
}


class GPUDecimate(bpy.types.Operator):

    bl_idname = "object.pamo"
    bl_label = "GPU Decimate"
    bl_info = "Decimate an manifold, intersect-free triangle mesh on GPU"
    bl_options = {'REGISTER', 'UNDO'}

    threshold_prop: bpy.props.FloatProperty(
        name="Threshold",
        default=0.1,
        min=0.0,
        soft_max=10.0
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        if obj.type != 'MESH':
            return False

        mesh = obj.data
        is_triangle_mesh = all(len(poly.vertices) == 3 for poly in mesh.polygons)
        return is_triangle_mesh

    def execute(self, context):
        obj = context.active_object
        mesh = obj.data
        v = numpy.empty(len(mesh.vertices) * 3, dtype=numpy.float32)
        t = numpy.empty(len(mesh.polygons) * 3, dtype=numpy.int32)
        mesh.vertices.foreach_get('co', v)
        mesh.polygons.foreach_get('vertices', t)
        v = v.reshape(-1, 3)
        t = t.reshape(-1, 3)
        vo = numpy.zeros_like(v)
        to = numpy.zeros_like(t)
        sh = numpy.array([len(v), len(t)], dtype=numpy.int32)
        s = self.threshold_prop / 1000
        cusimp.pamo_simplify(v.ctypes, t.ctypes, vo.ctypes, to.ctypes, sh.ctypes, s)
        mesh = bpy.data.meshes.new(name=mesh.name)
        n_v, n_t = sh
        mesh.vertices.add(n_v)
        mesh.vertices.foreach_set("co", vo[:n_v].ravel())
        mesh.loops.add(n_t * 3)
        mesh.loops.foreach_set("vertex_index", to[:n_t].ravel())
        mesh.update()
        mesh.polygons.add(n_t)
        mesh.polygons.foreach_set("loop_start", numpy.arange(n_t) * 3)
        mesh.polygons.foreach_set("loop_total", numpy.ones(n_t, dtype=numpy.int32) * 3)
        mesh.update()
        mesh.validate()
        obj.data = mesh
        return {'FINISHED'}


cusimp = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cusimp.dll"))

# float* in_verts, int* in_tris, float* out_verts, int* out_tris, int* in_out_nv_ni, float threshold
cusimp.pamo_simplify.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float)
cusimp.pamo_simplify.restype = None


def menu_func(self, context):
    self.layout.operator(GPUDecimate.bl_idname, icon='MOD_DECIM')


def register():
    bpy.utils.register_class(GPUDecimate)
    bpy.types.VIEW3D_MT_object_context_menu.append(menu_func)

def unregister():
    bpy.utils.unregister_class(GPUDecimate)
    bpy.types.VIEW3D_MT_object_context_menu.remove(menu_func)


if __name__ == "__main__":
    register()
