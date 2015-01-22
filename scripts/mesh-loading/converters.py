import Blender
import os
import bpy
import mathutils
from bpy.props import (BoolProperty,
    FloatProperty,
    StringProperty,
    EnumProperty,
    )
from bpy_extras.io_utils import (ImportHelper,
    ExportHelper,
    unpack_list,
    unpack_face_list,
    axis_conversion,
    )

def import_obj(path):
        Blender.Window.WaitCursor(1)
        name = path.split('\\')[-1].split('/')[-1]
        mesh = Blender.NMesh.New( name ) # create a new mesh
        # parse the file
        file = open(path, 'r')
        for line in file:
                words = line.split()
                if len(words) == 0 or words[0].startswith('#'):
                        pass
                elif words[0] == 'v':
                        x, y, z = float(words[1]), float(words[2]), float(words[3])
                        mesh.verts.append(Blender.NMesh.Vert(x, y, z))
                elif words[0] == 'f':
                        faceVertList = []
                        for faceIdx in words[1:]:
                                faceVert = mesh.verts[int(faceIdx)-1]
                                faceVertList.append(faceVert)
                        newFace = Blender.NMesh.Face(faceVertList)
                        mesh.addFace(newFace)
        
        # link the mesh to a new object
        ob = Blender.Object.New('Mesh', name) # Mesh must be spelled just this--it is a specific type
        ob.link(mesh) # tell the object to use the mesh we just made
        scn = Blender.Scene.GetCurrent()
        for o in scn.getChildren():
                o.sel = 0
        
        scn.link(ob) # link the object to the current scene
        ob.sel= 1
        ob.Layers = scn.Layers
        Blender.Window.WaitCursor(0)
        Blender.Window.RedrawAll()

bl_info = {
    "name": "OFF format",
    "description": "Import-Export OFF, Import/export simple OFF mesh.",
    "author": "Alex Tsui",
    "version": (0, 2),
    "blender": (2, 69, 0),
    "location": "File > Import-Export",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.5/Py/"
                "Scripts/My_Script",
    "category": "Import-Export"}

class ImportOFF(bpy.types.Operator, ImportHelper):
    """Load an OFF Mesh file"""
    bl_idname = "import_mesh.off"
    bl_label = "Import OFF Mesh"
    filename_ext = ".off"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='-Z',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Y',
            )

    def execute(self, context):
        #from . import import_off

        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
        ))
        global_matrix = axis_conversion(from_forward=self.axis_forward,
            from_up=self.axis_up,
            ).to_4x4()

        mesh = load(self, context, **keywords)
        if not mesh:
            return {'CANCELLED'}

        scene = bpy.context.scene
        obj = bpy.data.objects.new(mesh.name, mesh)
        scene.objects.link(obj)
        scene.objects.active = obj
        obj.select = True

        obj.matrix_world = global_matrix

        scene.update()

        return {'FINISHED'}

class ExportOFF(bpy.types.Operator, ExportHelper):
    """Save an OFF Mesh file"""
    bl_idname = "export_mesh.off"
    bl_label = "Export OFF Mesh"
    filter_glob = StringProperty(
        default="*.off",
        options={'HIDDEN'},
    )
    check_extension = True
    filename_ext = ".off"

    axis_forward = EnumProperty(
            name="Forward",
            items=(('X', "X Forward", ""),
                   ('Y', "Y Forward", ""),
                   ('Z', "Z Forward", ""),
                   ('-X', "-X Forward", ""),
                   ('-Y', "-Y Forward", ""),
                   ('-Z', "-Z Forward", ""),
                   ),
            default='-Z',
            )
    axis_up = EnumProperty(
            name="Up",
            items=(('X', "X Up", ""),
                   ('Y', "Y Up", ""),
                   ('Z', "Z Up", ""),
                   ('-X', "-X Up", ""),
                   ('-Y', "-Y Up", ""),
                   ('-Z', "-Z Up", ""),
                   ),
            default='Y',
            )

    def execute(self, context):
        keywords = self.as_keywords(ignore=('axis_forward',
            'axis_up',
            'filter_glob',
            'check_existing',
        ))
        global_matrix = axis_conversion(to_forward=self.axis_forward,
            to_up=self.axis_up,
            ).to_4x4()
        keywords['global_matrix'] = global_matrix
        return save(self, context, **keywords)

def menu_func_import(self, context):
    self.layout.operator(ImportOFF.bl_idname, text="OFF Mesh (.off)")

def menu_func_export(self, context):
    self.layout.operator(ExportOFF.bl_idname, text="OFF Mesh (.off)")

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)
    bpy.types.INFO_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)

def load(filepath):
    # Parse mesh from OFF file
    # TODO: Add support for NOFF and COFF
    filepath = os.fsencode(filepath)
    file = open(filepath, 'r')
    file.readline()
    vcount, fcount, ecount = [int(x) for x in file.readline().split()]
    verts = []
    facets = []
    i=0;
    while i<vcount:
        line = file.readline()
        try:
            px, py, pz = [float(x) for x in line.split()]
        except ValueError:
            continue
        verts.append((px, py, pz))
        i=i+1

    i=0;
    while i<fcount:
        line = file.readline()
        try:
            unused, vid1, vid2, vid3 = [int(x) for x in line.split()]
        except ValueError:
            continue
        facets.append((vid1, vid2, vid3))
        i=i+1

    # Assemble mesh
    off_name = bpy.path.display_name_from_filepath(filepath)
    mesh = bpy.data.meshes.new(name=off_name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", unpack_list(verts))

    mesh.tessfaces.add(len(facets))
    mesh.tessfaces.foreach_set("vertices_raw", unpack_face_list(facets))

    mesh.validate()
    mesh.update()

    return mesh

def save(operator, context, filepath,
    global_matrix = None):
    # Export the selected mesh
    APPLY_MODIFIERS = True # TODO: Make this configurable
    if global_matrix is None:
        global_matrix = mathutils.Matrix()
    scene = context.scene
    obj = scene.objects.active
    mesh = obj.to_mesh(scene, APPLY_MODIFIERS, 'PREVIEW')

    # Apply the inverse transformation
    obj_mat = obj.matrix_world
    mesh.transform(global_matrix * obj_mat)

    verts = mesh.vertices[:]
    facets = [ f for f in mesh.tessfaces ]

    # Write geometry to file
    filepath = os.fsencode(filepath)
    fp = open(filepath, 'w')

    fp.write('OFF\n')
    fp.write('%d %d 0\n' % (len(verts), len(facets)))

    for vert in verts:
        fp.write('%.16f %.16f %.16f\n' % vert.co[:])

    for facet in facets:
        fp.write('%d' % len(facet.vertices))
        for vid in facet.vertices:
            fp.write(' %d' % vid)
        fp.write('\n')

    fp.close()

    return {'FINISHED'}

if __name__ == "__main__":
    register()

