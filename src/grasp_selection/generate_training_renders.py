import os
import sys
sys.path.append('/Users/derasagis/anaconda3/envs/h5serv/lib/python2.7/site-packages')

import database as db
import experiment_config as ec
import maya_renderer as mr
import rendered_image as ri
sys.path.append('feature_vectors/')
import mesh_database as md

if __name__ == '__main__':
    test_on_subset = True
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)
    render_mode = config['maya']['render_mode']
    save_images = config['maya']['save_images']

    renderer = mr.MayaRenderer(config)
    database_name = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_name, config, access_level=db.READ_WRITE_ACCESS)

    obj_database = md.SHRECObjectDatabase(config['mesh_database_file'])
    img_paths = []

    category_to_index = dict() 

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)
        for obj in dataset:
            if test_on_subset and len(category_to_index) > 100:
                print('breaking')
                break
            obj_category = obj_database.object_category_for_key(obj.key)
            if obj_category not in category_to_index.keys():
                category_to_index[obj_category] = len(category_to_index)
            stable_poses = dataset.stable_poses(obj.key)
            for i, stable_pose in enumerate(stable_poses):
                if stable_pose.p > config['maya']['min_prob']:
                    rendered_images = renderer.render(obj, dataset, render_mode=render_mode, rot=stable_pose.r, extra_key='_stp_%d'%(i), save_images=save_images)
                    dataset.store_rendered_images(obj.key, rendered_images, stable_pose_id=stable_pose.id, image_type=render_mode)
                    for image in rendered_images:
                        img_paths.append((image.image_file[len(config['maya']['dest_dir'])+1:], category_to_index[obj_category]))

    for path in img_paths:
        print('%s %s' % path)
   
    database.close()

