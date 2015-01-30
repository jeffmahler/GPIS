import os
import sys

#data_folder = '/mnt/terastation/shape_data/Cat50_ModelDatabase'
data_folder = sys.argv[1]

data_list = os.listdir(data_folder)

# walk through the shape directory structure
for root, sub_folders, files in os.walk(data_folder):
    for f in files:
        # create file name
        file_name = os.path.join(root, f)
        file_root, file_ext = os.path.splitext(file_name)

        # handle only ply files
        if file_ext == '.ply':
            meshlabserver_cmd = 'echo graspdvrk | sudo -s meshlabserver -i %s -o %s.obj' %(file_name, file_root) 
            print meshlabserver_cmd
            os.system(meshlabserver_cmd)
