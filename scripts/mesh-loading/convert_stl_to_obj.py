import os
import sys
import IPython

INVALID_CATEGORIES = ['mnt', 'terastation', 'shape_data', 'amazon_picking_challenge', 'textured_meshes', '']

def get_category(file_dir, dataset):
    if dataset == 'berkeley':
        head, tail = os.path.split(file_dir)
        while head != '/' and tail in INVALID_CATEGORIES:
            head, tail = os.path.split(head)
        return tail
    return None

format = '.stl'
data_folder = sys.argv[1]
out_folder = sys.argv[2]
dataset = sys.argv[3]
file_match = sys.argv[4]

data_list = os.listdir(data_folder)

# walk through the shape directory structure
for root, sub_folders, files in os.walk(data_folder):
    category = get_category(root, dataset)
    if category in INVALID_CATEGORIES:
        continue

    for f in files:
        # create file name
        file_name = os.path.join(root, f)
        file_head, file_root = os.path.split(file_name)
        file_root, file_ext = os.path.splitext(file_root)

        # handle only ply files
        if file_name.endswith(format) and file_name.find(file_match) >= 0:
            print 'Converting ', file_name
            if dataset == 'berkeley':
                new_file_name = os.path.join(out_folder, category + '.obj')
            else:
                new_file_name = os.path.join(out_folder, file_root + '.obj')
            meshlabserver_cmd = 'meshlabserver -i %s -o %s' %(file_name, new_file_name) 
            print meshlabserver_cmd
            os.system(meshlabserver_cmd)
