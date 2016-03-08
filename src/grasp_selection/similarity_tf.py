
import numbers
import numpy as np
import tfx

import IPython

class SimilarityTransform3D:
    def __init__(self, pose=tfx.pose(np.eye(4)), scale=1.0, from_frame='world', to_frame='my_frame'):
        if not isinstance(pose, tfx.canonical.CanonicalTransform):
            raise ValueError('Pose must be tfx canonical tf')
        self.pose_ = pose
        self.scale_ = scale
        self.from_frame_ = from_frame
        self.to_frame_ = to_frame
        
    def apply(self, x, direction = False):
        """ Applies a similarity transform to a point x"""
        # allow numpy arays
        if isinstance(x, np.ndarray) and x.shape[0] == 3:
            num_pts = 1
            if x.ndim > 1:
                num_pts = x.shape[1]
            else:
                x = np.array([x]).T

            # rotation only if a direction
            if direction:
                x_tf = np.array(self.pose_.rotation.matrix.dot(x))
                if num_pts == 1:
                    x_tf = x_tf.squeeze()
                return x_tf

            # multiply pose matrix and scale result
            x_homog = np.r_[x, np.ones([1, num_pts])]
            x_homog_tf = np.array(self.pose_.matrix.dot(x_homog))
            x_tf = x_homog_tf[0:3,:]
            x_tf_scale = (1.0 / self.scale_) * x_tf
            x_tf_scale = np.array(x_tf_scale)
            if num_pts == 1:
                x_tf_scale = x_tf_scale.squeeze()

            return x_tf_scale 
        elif (isinstance(x, np.ndarray) and x.shape[0] == 1) or isinstance(x, numbers.Number):
            x_tf = (1.0 / self.scale_) * x
            return x_tf
        else:
            raise ValueError('Only numpy 3-arrays are supported')

    def dot(self, other_tf):
        """ Compose this transform with another """
        if other_tf.to_frame != self.from_frame:
            raise ValueError('Output reference frame of right hand side must match input reference frame of left hand side')
        pose_tf = self.pose_.matrix.dot(other_tf.pose.matrix)
        scale_tf = self.scale_ * other_tf.scale
        return SimilarityTransform3D(pose=tfx.pose(pose_tf), scale=scale_tf, from_frame=other_tf.from_frame, to_frame=self.to_frame)

    def inverse(self):
        """ Take the inverse of the transform """
        inv_pose = self.pose_.inverse()
        inv_pose.position = (1.0 / self.scale_) * inv_pose.position
        return SimilarityTransform3D(inv_pose, 1.0 / self.scale_, from_frame=self.to_frame, to_frame=self.from_frame)

    @staticmethod
    def load(filename):
        """ Load the transform """
        f = open(filename, 'r')
        line_list = list(f)
        from_frame= line_list[0][:-1]
        to_frame = line_list[1][:-1]
        scale = float(line_list[2][:-1])

        t_tokens = line_list[3][:-1].split()
        t = np.zeros(3)
        t[0] = float(t_tokens[0])
        t[1] = float(t_tokens[1])
        t[2] = float(t_tokens[2])

        R = np.zeros([3,3])
        r_tokens = line_list[4][:-1].split()
        R[0,0] = float(r_tokens[0])
        R[0,1] = float(r_tokens[1])
        R[0,2] = float(r_tokens[2])

        r_tokens = line_list[5][:-1].split()
        R[1,0] = float(r_tokens[0])
        R[1,1] = float(r_tokens[1])
        R[1,2] = float(r_tokens[2])

        r_tokens = line_list[6][:-1].split()
        R[2,0] = float(r_tokens[0])
        R[2,1] = float(r_tokens[1])
        R[2,2] = float(r_tokens[2])
        pose = tfx.pose(R, t)
        return SimilarityTransform3D(pose=pose, scale=scale, from_frame=from_frame, to_frame=to_frame)

    def save(self, filename):
        """ Save the transform to a custom file format """
        R = self.rotation
        t = self.translation
        f = open(filename, 'w')
        f.write('%s\n' %(self.from_frame))
        f.write('%s\n' %(self.to_frame))
        f.write('%f\n' %(self.scale))
        f.write('%f %f %f\n' %(t[0], t[1], t[2]))
        f.write('%f %f %f\n' %(R[0,0], R[0,1], R[0,2]))
        f.write('%f %f %f\n' %(R[1,0], R[1,1], R[1,2]))
        f.write('%f %f %f\n' %(R[2,0], R[2,1], R[2,2]))
        f.close()

    def save_pose_csv(self, filename):
        """ Save only the pose matrix to file """
        np.savetxt(filename, np.array(self.pose.matrix), delimiter=',')

    @property
    def translation(self):
        return np.array(self.pose_.position).squeeze()

    @property
    def rotation(self):
        return np.array(self.pose_.rotation.matrix)

    @property
    def pose(self):
        return self.pose_

    @property
    def from_frame(self):
        return self.from_frame_

    @property
    def to_frame(self):
        return self.to_frame_

    @pose.setter
    def pose(self, pose):
        self.pose_ = pose

    @property
    def scale(self):
        return self.scale_

    @scale.setter
    def scale(self, scale):
        self.scale_ = scale
