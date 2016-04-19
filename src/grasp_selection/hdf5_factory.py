"""
A bunch of classes for converting hdf5 groups & datasets to common object types
Author: Jeff Mahler
"""
import database as db
import features as f
import feature_functions as ff
import grasp
import mesh
import rendered_image as ri
import sdf
import similarity_tf as stf
import stable_pose_class as stpc

import datetime as dt
import h5py
import IPython
import logging
import numpy as np
import tfx

# data keys for easy access
SDF_DATA_KEY = 'data'
SDF_ORIGIN_KEY = 'origin'
SDF_RES_KEY = 'resolution'
SDF_POSE_KEY = 'pose'
SDF_SCALE_KEY = 'scale'
SDF_FRAME_KEY = 'frame'

MESH_VERTICES_KEY = 'vertices'
MESH_TRIANGLES_KEY = 'triangles'
MESH_NORMALS_KEY = 'normals'
MESH_POSE_KEY = 'pose'
MESH_SCALE_KEY = 'scale'
MESH_DENSITY_KEY = 'density'

LOCAL_FEATURE_NUM_FEAT_KEY = 'num_features'
LOCAL_FEATURE_DESC_KEY = 'descriptors'
LOCAL_FEATURE_RF_KEY = 'rfs'
LOCAL_FEATURE_POINT_KEY = 'points'
LOCAL_FEATURE_NORMAL_KEY = 'normals'
SHOT_FEATURES_KEY = 'shot'
FEATURE_KEY = 'feature'

NUM_STP_KEY = 'num_stable_poses'
POSE_KEY = 'pose'
STABLE_POSE_PROB_KEY = 'p'
STABLE_POSE_ROT_KEY = 'r'
STABLE_POSE_PT_KEY = 'x0'

NUM_GRASPS_KEY = 'num_grasps'
GRASP_KEY = 'grasp'
GRASP_ID_KEY = 'id'
GRASP_TYPE_KEY = 'type'
GRASP_CONFIGURATION_KEY = 'configuration'
GRASP_RF_KEY = 'frame'
GRASP_TIMESTAMP_KEY = 'timestamp'
GRASP_METRICS_KEY = 'metrics'
GRASP_FEATURES_KEY = 'features'

GRASP_FEATURE_NAME_KEY = 'name'
GRASP_FEATURE_TYPE_KEY = 'type'
GRASP_FEATURE_VECTOR_KEY = 'vector'

NUM_IMAGES_KEY = 'num_images'
IMAGE_KEY = 'image'
IMAGE_DATA_KEY = 'image_data'
CAM_POS_KEY = 'cam_pos'
CAM_ROT_KEY = 'cam_rot'
CAM_INT_PT_KEY = 'cam_int_pt'

class Hdf5ObjectFactory(object):
    """ Factory for spawning off new objects from HDF5 fields """

    @staticmethod
    def sdf_3d(data):
        """ Converts HDF5 data provided in dictionary |data| to an SDF object """
        sdf_data = np.array(data[SDF_DATA_KEY])
        origin = np.array(data.attrs[SDF_ORIGIN_KEY])
        resolution = data.attrs[SDF_RES_KEY]
        
        # should never be used, really
        pose = tfx.identity_tf()
        scale = 1.0
        if SDF_POSE_KEY in data.attrs.keys():
            pose = data.attrs[SDF_POSE_KEY]
        if SDF_SCALE_KEY in data.attrs.keys():
            scale = data.attrs[SDF_SCALE_KEY]
        tf = stf.SimilarityTransform3D(pose, scale)

        frame = None
        if SDF_FRAME_KEY in data.attrs.keys():
            frame = data.attrs[SDF_FRAME_KEY]
        
        return sdf.Sdf3D(sdf_data, origin, resolution, tf, frame)

    @staticmethod
    def write_sdf_3d(sdf, data):
        """ Writes sdf object to HDF5 data provided in |data| """
        data.create_dataset(SDF_DATA_KEY, data=sdf.data)
        data.attrs.create(SDF_ORIGIN_KEY, sdf.origin)
        data.attrs.create(SDF_RES_KEY, sdf.resolution)
        
    @staticmethod
    def mesh_3d(data):
        """ Converts HDF5 data provided in dictionary |data| to a mesh object """
        vertices = np.array(data[MESH_VERTICES_KEY])
        triangles = np.array(data[MESH_TRIANGLES_KEY])

        normals = None
        if MESH_NORMALS_KEY in data.keys():
            normals = np.array(data[MESH_NORMALS_KEY])
        
        # should never be used, really
        pose = tfx.identity_tf()
        scale = 1.0
        if MESH_POSE_KEY in data.attrs.keys():
            pose = data.attrs[MESH_POSE_KEY]
        if MESH_SCALE_KEY in data.attrs.keys():
            scale = data.attrs[MESH_SCALE_KEY]

        density = 1.0
        if MESH_DENSITY_KEY in data.attrs.keys():
            density = data.attrs[MESH_DENSITY_KEY]            
        
        metadata = None # really should not have been associated with Meshes in the first place
        return mesh.Mesh3D(vertices.tolist(), triangles.tolist(), normals=normals, metadata=metadata, pose=pose, scale=scale, density=density)

    @staticmethod
    def write_mesh_3d(mesh, data, force_overwrite=False):
        """ Writes mesh object to HDF5 data provided in |data| """
        if MESH_VERTICES_KEY in data.keys() or MESH_TRIANGLES_KEY in data.keys():
            if not force_overwrite:
                logging.warning('Mesh already exists and force overwrite not specified. Aborting')
                return False

            del data[MESH_VERTICES_KEY]
            del data[MESH_TRIANGLES_KEY]
            if mesh.normals() is not None:
                del data[MESH_NORMALS_KEY]

        data.create_dataset(MESH_VERTICES_KEY, data=np.array(mesh.vertices()))
        data.create_dataset(MESH_TRIANGLES_KEY, data=np.array(mesh.triangles()))
        if mesh.normals() is not None:
            data.create_dataset(MESH_NORMALS_KEY, data=np.array(mesh.normals()))
        return True

    @staticmethod
    def local_features(data):
        """ Converts HDF5 data provided in dictionary |data| to a local feature object """
        features = f.BagOfFeatures()
        num_features = data.attrs[LOCAL_FEATURE_NUM_FEAT_KEY]
        descriptors = data[LOCAL_FEATURE_DESC_KEY]
        rfs = data[LOCAL_FEATURE_RF_KEY]
        keypoints = data[LOCAL_FEATURE_POINT_KEY]
        normals = data[LOCAL_FEATURE_NORMAL_KEY]

        for i in range(num_features):
            features.add(f.LocalFeature(descriptors[i,:], rfs[i,:], keypoints[i,:], normals[i,:]))            
        return features

    @staticmethod
    def write_shot_features(local_features, data):
        """ Writes shot features to HDF5 data provided in |data| """
        data.create_group(SHOT_FEATURES_KEY)
        data[SHOT_FEATURES_KEY].attrs.create(LOCAL_FEATURE_NUM_FEAT_KEY, local_features.num_features)
        data[SHOT_FEATURES_KEY].create_dataset(LOCAL_FEATURE_DESC_KEY, data=local_features.descriptors)
        data[SHOT_FEATURES_KEY].create_dataset(LOCAL_FEATURE_RF_KEY, data=local_features.reference_frames)
        data[SHOT_FEATURES_KEY].create_dataset(LOCAL_FEATURE_POINT_KEY, data=local_features.keypoints)
        data[SHOT_FEATURES_KEY].create_dataset(LOCAL_FEATURE_NORMAL_KEY, data=local_features.normals)

    @staticmethod
    def stable_poses(data):
        """ Read out a list of stable pose objects """
        num_stable_poses = data.attrs[NUM_STP_KEY]
        stable_poses = []
        for i in range(num_stable_poses):
            stp_key = POSE_KEY + '_' + str(i) 
            p = data[stp_key].attrs[STABLE_POSE_PROB_KEY]
            r = data[stp_key].attrs[STABLE_POSE_ROT_KEY]
            try:
                x0 = data[stp_key].attrs[STABLE_POSE_PT_KEY]
            except:
                x0 = np.zeros(3)
            stable_poses.append(stpc.StablePose(p, r, x0, stp_id=stp_key))
        return stable_poses

    @staticmethod
    def stable_pose(data, stable_pose_id):
        """ Read out a stable pose object """
        p = data[stable_pose_id].attrs[STABLE_POSE_PROB_KEY]
        r = data[stable_pose_id].attrs[STABLE_POSE_ROT_KEY]
        try:
            x0 = data[stable_pose_id].attrs[STABLE_POSE_PT_KEY]
        except:
            x0 = np.zeros(3)
        return stpc.StablePose(p, r, x0, stp_id=stable_pose_id)

    @staticmethod
    def write_stable_poses(stable_poses, data):
        """ Writes shot features to HDF5 data provided in |data| """
        num_stable_poses = len(stable_poses)
        data.attrs.create(NUM_STP_KEY, num_stable_poses)
        for i, stable_pose in enumerate(stable_poses):
            stp_key = POSE_KEY + '_' + str(i)
            data.create_group(stp_key)
            data[stp_key].attrs.create(STABLE_POSE_PROB_KEY, stable_pose.p)
            data[stp_key].attrs.create(STABLE_POSE_ROT_KEY, stable_pose.r)
            data[stp_key].attrs.create(STABLE_POSE_PT_KEY, stable_pose.x0)
            data[stp_key].create_group(db.RENDERED_IMAGES_KEY)

    @staticmethod
    def grasps(data):
        """ Return a list of grasp objects from the data provided in the HDF5 dictionary """
        # need to read in a bunch of grasps but also need to know what kind of grasp it is
        grasps = []
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        for i in range(num_grasps):
            # get the grasp data y'all
            grasp_key = GRASP_KEY + '_' + str(i)
            if grasp_key in data.keys():
                grasp_id =      data[grasp_key].attrs[GRASP_ID_KEY]            
                grasp_type =    data[grasp_key].attrs[GRASP_TYPE_KEY]
                configuration = data[grasp_key].attrs[GRASP_CONFIGURATION_KEY]
                frame =         data[grasp_key].attrs[GRASP_RF_KEY]            
                timestamp =     data[grasp_key].attrs[GRASP_TIMESTAMP_KEY]            
                
                # create object based on type
                g = None
                if grasp_type == 'ParallelJawPtGrasp3D':
                    g = grasp.ParallelJawPtGrasp3D(configuration=configuration, frame=frame, timestamp=timestamp, grasp_id=grasp_id)
                grasps.append(g)
            else:
                logging.debug('Grasp %s is corrupt. Skipping' %(grasp_key))

        return grasps

    @staticmethod
    def write_grasps(grasps, data, force_overwrite=False):
        """ Writes shot features to HDF5 data provided in |data| """
        num_grasps = data.attrs[NUM_GRASPS_KEY]
        num_new_grasps = len(grasps)

        # get timestamp for pruning old grasps
        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' %(dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 

        # add each grasp
        for i, grasp in enumerate(grasps):
            grasp_id = grasp.grasp_id
            if grasp_id is None:
                grasp_id = i+num_grasps
            grasp_key = GRASP_KEY + '_' + str(grasp_id)

            if grasp_key not in data.keys():
                data.create_group(grasp_key)
                data[grasp_key].attrs.create(GRASP_ID_KEY, grasp_id)
                data[grasp_key].attrs.create(GRASP_TYPE_KEY, type(grasp).__name__)
                data[grasp_key].attrs.create(GRASP_CONFIGURATION_KEY, grasp.configuration)
                data[grasp_key].attrs.create(GRASP_RF_KEY, grasp.frame)
                data[grasp_key].attrs.create(GRASP_TIMESTAMP_KEY, creation_stamp)
                data[grasp_key].create_group(GRASP_METRICS_KEY) 
                data[grasp_key].create_group(GRASP_FEATURES_KEY)
            elif force_overwrite:
                data[grasp_key].attrs[GRASP_ID_KEY] = grasp_id
                data[grasp_key].attrs[GRASP_TYPE_KEY] = type(grasp).__name__
                data[grasp_key].attrs[GRASP_CONFIGURATION_KEY] = grasp.configuration
                data[grasp_key].attrs[GRASP_RF_KEY] = grasp.frame
                data[grasp_key].attrs[GRASP_TIMESTAMP_KEY] = creation_stamp
            else:
                logging.warning('Grasp %d already exists and overwrite was not requested. Aborting write request' %(grasp_id))
                return None

        data.attrs[NUM_GRASPS_KEY] = num_grasps + num_new_grasps
        return creation_stamp

    @staticmethod
    def grasp_metrics(grasps, data):
        """ Returns a dictionary of the metrics for the given grasps """
        grasp_metrics = {}
        for grasp in grasps:
            grasp_id = grasp.grasp_id
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            grasp_metrics[grasp_id] = {}
            if grasp_key in data.keys():
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]                
                for metric_name in grasp_metric_data.attrs.keys():
                    grasp_metrics[grasp_id][metric_name] = grasp_metric_data.attrs[metric_name]
        return grasp_metrics

    @staticmethod
    def write_grasp_metrics(grasp_metric_dict, data, force_overwrite=False):
        """ Write grasp metrics to database """
        for grasp_id, metric_dict in grasp_metric_dict.iteritems():
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            if grasp_key in data.keys():
                grasp_metric_data = data[grasp_key][GRASP_METRICS_KEY]

                for metric_tag, metric in metric_dict.iteritems():
                    if metric_tag not in grasp_metric_data.attrs.keys():
                        grasp_metric_data.attrs.create(metric_tag, metric)
                    elif force_overwrite:
                        grasp_metric_data.attrs[metric_tag] = metric
                    else:
                        logging.warning('Metric %s already exists for grasp %s and overwrite was not requested. Aborting write request' %(metric_tag, grasp_id))
                        return False
        return True

    @staticmethod
    def grasp_features(grasps, data):
        """ Read grasp features from database into a dictionary """
        features = {}
        for grasp in grasps:
            grasp_id = grasp.grasp_id
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            features[grasp_id] = []
            if grasp_key in data.keys():
                grasp_feature_data = data[grasp_key][GRASP_FEATURES_KEY] 
                for feature_name in grasp_feature_data.keys():
                    features[grasp_id].append(ff.GraspFeature(feature_name, grasp_feature_data[feature_name].attrs[GRASP_FEATURE_TYPE_KEY],
                                                              grasp_feature_data[feature_name].attrs[GRASP_FEATURE_VECTOR_KEY]))
        return features

    @staticmethod
    def write_grasp_features(grasp_feature_dict, data, force_overwrite=False):
        """ Write grasp metrics to database """
        for grasp_id, feature_list in grasp_feature_dict.iteritems():
            grasp_key = GRASP_KEY + '_' + str(grasp_id)
            if grasp_key in data.keys():
                # parse all feature extractor objects
                for feature in feature_list:
                    if feature.name not in data[grasp_key][GRASP_FEATURES_KEY].keys():
                        data[grasp_key][GRASP_FEATURES_KEY].create_group(feature.name)
                        data[grasp_key][GRASP_FEATURES_KEY][feature.name].attrs.create(GRASP_FEATURE_TYPE_KEY, feature.typename)
                        data[grasp_key][GRASP_FEATURES_KEY][feature.name].attrs.create(GRASP_FEATURE_VECTOR_KEY, feature.descriptor) # save the feature vector, class should know how to back out the relevant variables
                    elif force_overwrite:
                        data[grasp_key][GRASP_FEATURES_KEY][feature.name].attrs[GRASP_FEATURE_TYPE_KEY] = feature.typename
                        data[grasp_key][GRASP_FEATURES_KEY][feature.name].attrs[GRASP_FEATURE_VECTOR_KEY] = feature.descriptor # save the feature vector, class should know how to back out the relevant variables
                    else:
                        logging.warning('Feature %s already exists for grasp %s and overwrite was not requested. Aborting write request' %(feature_tag, grasp_id))
                        return False
        return True

    @staticmethod
    def rendered_images(data):
        rendered_images = []
        num_images = data.attrs[NUM_IMAGES_KEY]
        
        for i in range(num_images):
            # get the image data y'all
            image_key = IMAGE_KEY + '_' + str(i)
            image_data = data[image_key]
            image =           np.array(image_data[IMAGE_DATA_KEY])       
            cam_pos =         image_data.attrs[CAM_POS_KEY]
            cam_rot =         image_data.attrs[CAM_ROT_KEY]
            cam_interest_pt = image_data.attrs[CAM_INT_PT_KEY]            

            rendered_images.append(ri.RenderedImage(image, cam_pos, cam_rot, cam_interest_pt, image_id=i))
        return rendered_images

    @staticmethod
    def write_rendered_images(rendered_images, data, force_overwrite=False):
        """ Write rendered images to database """
        num_images = 0
        if NUM_IMAGES_KEY in data.keys():
            num_images = data.attrs[NUM_GRASPS_KEY]
        num_new_images = len(rendered_images)        

        for image_id, rendered_image in enumerate(rendered_images):
            image_key = IMAGE_KEY + '_' + str(image_id)
            if image_key not in data.keys():
                data.create_group(image_key)
                image_data = data[image_key]

                image_data.create_dataset(IMAGE_DATA_KEY, data=rendered_image.image)
                image_data.attrs.create(CAM_POS_KEY, rendered_image.cam_pos)
                image_data.attrs.create(CAM_ROT_KEY, rendered_image.cam_rot)
                image_data.attrs.create(CAM_INT_PT_KEY, rendered_image.cam_interest_pt)
            elif force_overwrite:
                image_data[IMAGE_DATA_KEY] = rendered_image.image
                image_data.attrs[CAM_POS_KEY] = rendered_image.cam_pos
                image_data.attrs[CAM_ROT_KEY] = rendered_image.cam_rot
                image_data.attrs[CAM_INT_PT_KEY] = rendered_image.cam_interest_pt
            else:
                logging.warning('Image %d already exists and overwrite was not requested. Aborting write request' %(image_id))
                return None

        if NUM_IMAGES_KEY in data.keys():
            data.attrs[NUM_IMAGES_KEY] = num_images + num_new_images
        else:
            data.attrs.create(NUM_IMAGES_KEY, num_images + num_new_images)
            
