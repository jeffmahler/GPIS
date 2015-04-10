import os
import sys

import obj_file
import mesh

argc = len(sys.argv)
data_folder = sys.argv[1]

min_dim = -1
if argc > 2:
    min_dim = float(sys.argv[2])

data_list = os.listdir(data_folder)

# walk through the shape directory structure
for root, sub_folders, files in os.walk(data_folder):
    for f in files:
        # create file name
        file_name = os.path.join(root, f)
        file_root, file_ext = os.path.splitext(file_name)

        # handle only ply files
        if file_ext == '.obj' and file_name.find("_clean") == -1:
            print 'Cleaning ', file_name

            # read the mesh
            of = obj_file.ObjFile(file_name)
            m = of.read()
            
            # clean up mesh
            try:
                m.remove_unreferenced_vertices()
                m.normalize_vertices()

                if min_dim > 0:
                    m.rescale_vertices(min_dim)

                # write new file
                clean_file_name = '%s_clean.obj' %(file_root)
                oof = obj_file.ObjFile(clean_file_name)
                oof.write(m)
            except ValueError:
                print 'Failed on', file_name
