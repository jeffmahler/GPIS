
import numbers
import numpy as np
import tfx

import IPython

class SimilarityTransform3D:
    def __init__(self, pose, scale=1.0):
        if not isinstance(pose, tfx.canonical.CanonicalTransform):
            raise ValueError('Pose must be tfx canonical tf')
        self.pose_ = pose
        self.scale_ = scale
        
    def apply(self, x, normalize = False):
        """ Appliess a similarity transform to a point x"""
        # allow numpy arays
        if isinstance(x, np.ndarray) and x.shape[0] == 3:
            num_pts = 1
            if x.ndim > 1:
                num_pts = x.shape[1]
            else:
                x = np.array([x]).T
            x_homog = np.r_[x, np.ones([1, num_pts])]
            x_homog_tf = self.pose_.matrix.dot(x_homog)
            x_tf = x_homog_tf[0:3,:]
            x_tf_scale = (1.0 / self.scale_) * x_tf

            x_tf_scale = np.array(x_tf_scale)
            if num_pts == 1:
                x_tf_scale = x_tf_scale.squeeze()

            if normalize:
                if num_pts == 1:
                    x_tf_scale = x_tf_scale / np.linalg.norm(x_tf_scale)
                else:
                    x_tf_scale = x_tf_scale / np.tile(np.linalg.norm(x_tf_scale, axis=0), [3, 1])

            return x_tf_scale 
        elif (isinstance(x, np.ndarray) and x.shape[0] == 1) or isinstance(x, numbers.Number):
            x_tf = (1.0 / self.scale_) * x
            return x_tf
        else:
            raise ValueError('Only numpy 3-arrays are supported')

    def inverse(self):
        inv_pose = self.pose_.inverse()
        inv_pose.position = (1.0 / self.scale_) * inv_pose.position
        return SimilarityTransform3D(inv_pose, 1.0 / self.scale_)

    @property
    def pose(self):
        return self.pose_

    @pose.setter
    def pose(self, pose):
        self.pose_ = pose

    @property
    def scale(self):
        return self.scale_

    @scale.setter
    def scale(self, scale):
        self.scale_ = scale
