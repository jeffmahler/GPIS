import os
import shutil
import sys
import IPython

import obj_file

# global vars
SHAPE_DB_ROOT = '/mnt/terastation/shape_data'
DATASETS = ['amazon_picking_challenge', 'Archive3D', 'BigBIRD', 'Cat50_ModelDatabase', 'KIT', 'ModelNet40', 'NTU3D',
            'PrincetonShapeBenchmark', 'SHREC14LSGTB', 'YCB']
DATASET_EXTENSIONS = ['.obj', '.3DS', '.obj', '.obj', '.obj', '.off', '.obj', '.off', '.off', '.obj']
DATASET_FILTERS = ['poisson_texture_mapped', '', 'poisson_texture_mapped', '', '800_tex', '', '', '', '', 'poisson_texture_mapped']
DATASET_FIX_NAMES = [True, False, True, False, False, False, False, False, False, True]
DATASET_SCALES = [1, 1, 1, 1, 1e-2, 1, 1, 1, 1, 1] 

INVALID_CATEGORIES = ['mnt', 'terastation', 'shape_data', 'textured_meshes', 'processed', '']
INVALID_CATEGORIES.extend(DATASETS)

def get_category_from_directory(file_dir):
    """ Grabs the category from the filename """
    head, tail = os.path.split(file_dir)
    while head != '/' and tail in INVALID_CATEGORIES:
        head, tail = os.path.split(head)
    return tail

# read in params
argc = len(sys.argv)
dataset = sys.argv[1]
dest_root_folder = sys.argv[2]

# numeric params
min_dim = -1
if argc > 3:
    min_dim = float(sys.argv[3])

dim = 25
if argc > 4:
    dim = int(sys.argv[4])

padding = 5
if argc > 5:
    padding = int(sys.argv[5])

# create dataset folder path
if dataset != 'all' and dataset in DATASETS:
    dataset_inds = [DATASETS.index(dataset)]
elif dataset == 'all':
    dataset_inds = range(len(DATASETS))
else:
    raise Exception('Invalid dataset!')

# create dest folder if doesn't exist
if not os.path.exists(dest_root_folder):
    os.mkdir(dest_root_folder)

all_filenames = []
all_categories = []

# loop through datasets and cleanup
for j in range(len(dataset_inds)):
    # get next dataset
    i = dataset_inds[j]
    dataset = DATASETS[i]
    data_folder = os.path.join(SHAPE_DB_ROOT, dataset)
    data_list = os.listdir(data_folder)
    print 'Processing dataset', dataset

    # file extensions
    dataset_file_ext = DATASET_EXTENSIONS[i]
    dataset_filter = DATASET_FILTERS[i]
    dataset_fix_names = DATASET_FIX_NAMES[i]
    dataset_scale = DATASET_SCALES[i]
    print 'SCALE', dataset_scale

    # create target dir name
    target_dir = os.path.join(dest_root_folder, dataset)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        os.system('chmod a+rwx %s' %(target_dir) )

    # walk through the shape directory structure
    for root, sub_folders, files in os.walk(data_folder):
        category = get_category_from_directory(root)
        
        for f in files:
            # create file name
            filename = os.path.join(root, f)
#            f = 'Co.obj'
#            root = '/mnt/terastation/shape_data/Cat50_ModelDatabase/shoe'
#            filename = '/mnt/terastation/shape_data/Cat50_ModelDatabase/shoe/Co.obj'
            file_root, file_ext = os.path.splitext(f)
            fullpath_file_root = os.path.join(root, file_root)

            # grab the models with a given format and filtered by the given filter (fresh clean for each file)
            if file_ext == dataset_file_ext and filename.find(dataset_filter) > -1 and filename.find('clean') == -1:
                # convert to obj
                obj_filename = '%s.obj' %(fullpath_file_root)

                meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(filename, obj_filename) 
                os.system(meshlabserver_cmd)
                print 'MeshlabServer Command:', meshlabserver_cmd

                if not os.path.exists(obj_filename):
                    print 'Meshlab conversion failed for', obj_filename
                    continue

                # now load and cleanup
                print 'Cleaning ', filename
                of = obj_file.ObjFile(obj_filename)
                m = of.read()

                # get the category?
                if dataset_fix_names:
                    target_obj_filename = os.path.join(target_dir, category + '.obj')
                    target_sdf_filename = os.path.join(target_dir, category + '.sdf')
                else:
                    target_obj_filename = os.path.join(target_dir, file_root + '.obj')
                    target_sdf_filename = os.path.join(target_dir, file_root + '.sdf')

                # clean up mesh
                try:
                    m.remove_bad_tris()
                    m.remove_unreferenced_vertices()
                    m.normalize_vertices()

                    if min_dim > 0:
                        m.rescale_vertices(min_dim)
                    elif dataset_scale != 1:
                        m.rescale_vertices(dataset_scale)

                    # write new file
                    oof = obj_file.ObjFile(target_obj_filename)
                    oof.write(m)
                    os.system('chmod a+rwx \"%s\"' %(target_obj_filename) )
                except ValueError as e:
                    print e
                    print 'Failed on', filename
                    continue

                # convert to sdf
                sdfgen_cmd = '/home/jmahler/Libraries/SDFGen/bin/SDFGen \"%s\" %d %d' %(target_obj_filename, dim, padding)
                os.system(sdfgen_cmd)
                print 'SDF Command', sdfgen_cmd

                if not os.path.exists(target_sdf_filename):
                    print 'SDF computation failed for', target_sdf_filename
                    continue
                os.system('chmod a+rwx \"%s\"' %(target_sdf_filename) )

                # add filename to master list
                if dataset_fix_names:
                    all_filenames.append(category)                    
                else:
                    all_filenames.append(file_root)
                all_categories.append(category)
 
    # save filenames to master list
    master_list_filename = os.path.join(target_dir, 'index.db')
    out_f = open(master_list_filename, 'w')
    for filename, category in zip(all_filenames, all_categories):
        out_f.write('%s %s\n' %(filename, category))
    out_f.close()
    os.system('chmod a+rwx %s' %(master_list_filename) )
