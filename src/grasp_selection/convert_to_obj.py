import os
import shutil
import sys
import IPython
import stable_poses

from mesh_file import MeshFile
from mesh_cleaner import MeshCleaner

# global vars
SHAPE_DB_ROOT = '/Users/MelRod/' #/mnt/terastation/shape_data'
DATASETS = ['amazon_picking_challenge', 'Archive3D', 'BigBIRD', 'Cat50_ModelDatabase', 'KIT', 'ModelNet40', 'NTU3D',
            'PrincetonShapeBenchmark', 'SHREC14LSGTB', 'YCB']
DATASET_EXTENSIONS = ['.obj', '.3DS', '.obj', '.obj', '.obj', '.off', '.obj', '.off', '.off', '.obj']
DATASET_FILTERS = ['poisson_texture_mapped', '', 'poisson_texture_mapped', '', '800_tex', '', '', '', '', 'poisson_texture_mapped']
DATASET_FIX_NAMES = [True, False, True, False, False, False, False, False, False, True]
DATASET_SCALES = [1, 1, 1, 1, 1e-2, 1, 1, 1, 1, 1] 

INVALID_CATEGORIES = ['mnt', 'terastation', 'shape_data', 'textured_meshes', 'processed', '']
INVALID_CATEGORIES.extend(DATASETS)

GRIPPER_SIZE = 0.10

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
                # get the category?
                if dataset_fix_names:
                    target_filename = os.path.join(target_dir, category)
                else:
                    target_filename = os.path.join(target_dir, file_root)

                # convert to obj
                obj_filename = '%s.obj' %(fullpath_file_root)
                smoothing_script = ''
                m = MeshFile.extract_mesh(filename, obj_filename, smoothing_script)

                # clean up mesh
                mesh_cleaner = MeshCleaner(m)
                mesh_cleaner.remove_bad_tris()
                mesh_cleaner.remove_unreferenced_vertices()
                mesh_cleaner.normalize_vertices()

                mesh_cleaner.rescale_vertices(GRIPPER_SIZE, rescaling_type=MeshCleaner.RescalingTypeMin)

                # save obj, sdf, etc.
                MeshFile.write_obj(m, target_filename)
                # MeshFile.write_sdf(m, target_filename, dim, padding)
                # MeshFile.write_stp(m, target_filename)
                # MeshFile.write_json(m, target_filename)
                # MeshFile.write_shot(m, target_filename)

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
