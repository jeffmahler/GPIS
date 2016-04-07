import IPython
import numpy as np
import tfx

import similarity_tf as stf

class RenderedImage:
    """ Class to encapculate data from rendered images from maya """

    def __init__(self, image, cam_pos, cam_rot, cam_interest_pt, image_id=-1, stable_pose=None, obj_key=None):
        self.image = image
        self.cam_pos = cam_pos
        self.cam_rot = cam_rot
        self.cam_interest_pt = cam_interest_pt
        self.id = image_id
        self.stable_pose = stable_pose
        self.obj_key = obj_key
        self.descriptors = {}

        # HACK: to fix stable pose bug
        if self.stable_pose is not None and np.abs(np.linalg.det(self.stable_pose.r) + 1) < 0.01:
            self.stable_pose.r[1,:] = -self.stable_pose.r[1,:]

    def object_to_stp_transform(self):
        T_obj_obj_p = tfx.pose(self.stable_pose.r).matrix
        return stf.SimilarityTransform3D(pose=tfx.pose(T_obj_obj_p), from_frame='obj', to_frame='stp')

    def stp_to_camera_transform(self):
        """ Returns the transformation from camera to the object stable pose """
        # setup variables
        camera_xyz_w = self.cam_pos
        camera_rot_w = self.cam_rot
        camera_int_pt_w = self.cam_interest_pt
        camera_xyz_obj_p = camera_xyz_w - camera_int_pt_w
        
        # get the distance from the camera to the world
        camera_dist_xy = np.linalg.norm(camera_xyz_w[:2])
        z = [0,0,np.linalg.norm(camera_xyz_w[:3])]

        # form the rotations about the x and z axis for the object on the tabletop
        theta = camera_rot_w[0] * np.pi / 180.0
        phi = -camera_rot_w[2] * np.pi / 180.0 + np.pi / 2.0
        camera_rot_obj_p_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                                       [np.sin(phi), np.cos(phi), 0],
                                       [0, 0, 1]])

        camera_rot_obj_p_x = np.array([[1, 0, 0],
                                       [0, np.cos(theta), -np.sin(theta)],
                                       [0, np.sin(theta), np.cos(theta)]])
        
        # form the full rotation matrix, swapping axes to match maya
        camera_md = np.array([[0, 1, 0],[1, 0, 0],[0,0,-1]])
        camera_rot_obj_p = camera_md.dot(camera_rot_obj_p_z.dot(camera_rot_obj_p_x))
        camera_rot_obj_p = camera_rot_obj_p.T
            
        # form the full object to camera transform
        T_obj_p_camera = tfx.pose(camera_rot_obj_p, z).matrix
        return stf.SimilarityTransform3D(pose=tfx.pose(T_obj_p_camera), from_frame='stp', to_frame='camera')
    
    def camera_to_object_transform(self):
        """ Returns the transformation from camera to object when the object is in the given stable pose """
        # form the full object to camera transform
        T_obj_p_camera = self.stp_to_camera_transform()
        T_obj_obj_p = self.object_to_stp_transform()
        T_obj_camera = T_obj_p_camera.dot(T_obj_obj_p)
        return T_obj_camera

