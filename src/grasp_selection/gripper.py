"""
Class to encapsulate robot grippers
Author: Jeff
"""
import json
import numpy as np
import obj_file
import os
import similarity_tf as stf
import sys

GRIPPER_MESH_FILENAME = 'gripper.obj'
GRIPPER_PARAMS_FILENAME = 'params.json'
T_MESH_GRIPPER_FILENAME = 'T_mesh_gripper.stf' 
T_GRASP_GRIPPER_FILENAME = 'T_grasp_gripper.stf' 

class RobotGripper():
    def __init__(self, name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper):
        """
        Init a robot gripper with a name, its mesh file, and two transforms:
           T_mesh_gripper - the tf from gripper frame to the mesh frame (for rendering)
           T_grasp_gripper - the tf from gripper frame to the canonical grasp frame (y-axis = grasp axis, x-axis = palm axis)
        The gripper frame should be the frame used to command the physical robot
        """
        self.name = name
        self.mesh = mesh
        self.mesh_filename = mesh_filename
        self.T_mesh_gripper = T_mesh_gripper
        self.T_grasp_gripper = T_grasp_gripper
        for key, value in params.iteritems():
            setattr(self, key, value)

    def collides_with_table(self, grasp, stable_pose, clearance = 0):
        """ Checks whether or not the gripper collides with the table in the stable pose """
        # transform mesh into object pose to check collisions with table
        T_gripper_obj = grasp.gripper_transform(self)
        T_mesh_obj = self.T_mesh_gripper.dot(T_gripper_obj)
        mesh_tf = self.mesh.transform(T_mesh_obj.inverse())
        
        # extract table
        n = stable_pose.r[2,:]
        x0 = stable_pose.x0

        # check all vertices for intersection with table
        for vertex in mesh_tf.vertices():
            v = np.array(vertex)
            if n.dot(v - x0) < clearance:
                return True
        return False

    @staticmethod
    def load(gripper_name, gripper_dir='data/grippers'):
        """ Load the gripper specified by gripper_name """
        mesh_filename = os.path.join(gripper_dir, gripper_name, GRIPPER_MESH_FILENAME)
        mesh = obj_file.ObjFile(mesh_filename).read()
        
        f = open(os.path.join(os.path.join(gripper_dir, gripper_name, GRIPPER_PARAMS_FILENAME)), 'r')
        params = json.load(f)

        T_mesh_gripper = stf.SimilarityTransform3D.load(os.path.join(gripper_dir, gripper_name, T_MESH_GRIPPER_FILENAME)) 
        T_grasp_gripper = stf.SimilarityTransform3D.load(os.path.join(gripper_dir, gripper_name, T_GRASP_GRIPPER_FILENAME)) 
        return RobotGripper(gripper_name, mesh, mesh_filename, params, T_mesh_gripper, T_grasp_gripper)
