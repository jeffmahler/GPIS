'''
Reads and writes sdfs to file
Author: Jeff Mahler
'''
import logging
import numpy as np
import os
import tfx

import matplotlib.pyplot as plt
import sdf
import IPython

class SdfFile:
    def __init__(self, file_name):
        self.file_name_ = file_name
        file_name, file_ext = os.path.splitext(self.file_name_)

        if file_ext == '.sdf':
            self.use_3d_ = True
        elif file_ext == '.csv':
            self.use_3d_ = False
        else:
            raise ValueError('Extension %s invalid for SDFs' %(file_ext))
    
    def read(self):
        '''
        Reads an SDF from file
        '''
        # read in basic params from file
        try:
            if self.use_3d_:
                return self._read_3d()
            else:
                return self._read_2d()

        except IOError:
            logging.error('Failed to open %s as an SDF'%(self.file_name_)) 
            return None

    def _read_3d(self):
        '''
        Reads a 3d SDF
        '''
        my_file = open(self.file_name_, 'r')
        nx, ny, nz = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
        ox, oy, oz = [float(i) for i in my_file.readline().split()]   #shape origin
        dims = np.array([nx, ny, nz])
        origin = np.array([ox, oy, oz])

        resolution = float(my_file.readline()) # resolution of the grid cells in original mesh coords
        sdf_data = np.zeros(dims)

        # loop through file, getting each value
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    sdf_data[i][j][k] = float(my_file.readline())
                    count += 1 
        my_file.close()
        return sdf.Sdf3D(sdf_data, origin, resolution)

    def _read_2d(self):
        '''
        Reads a 2d SDF from a CSV file
        '''
        if not os.path.exists(self.file_name_):
            raise IOError('File does not exist')

        sdf_data = np.loadtxt(self.file_name_, delimiter=',') 
        return sdf.Sdf2D(sdf_data)

    def write(self, sdf):
        '''
        Writes an SDF to file
        '''
        todo = 1

def test_3d():
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf = SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    logging.info('Loaded 3D SDF %s'  %(sdf_3d_file_name))
    logging.info('Dimension: x: %d, y: %d, z: %d' % sdf_3d.dimensions)
    logging.info('Origin: (%f, %f, %f)' %(sdf_3d.origin[0], sdf_3d.origin[1], sdf_3d.origin[2]))
    logging.info('Resolution: %f' % sdf_3d.resolution)

    # tests
    assert sdf_3d.dimensions[0] == 25
    assert sdf_3d.dimensions[1] == 25
    assert sdf_3d.dimensions[2] == 25

    assert sdf_3d.origin[0] == -1.04021
    assert sdf_3d.origin[1] == -1.1007
    assert sdf_3d.origin[2] == -1.01391

    assert sdf_3d.resolution == 0.0852112

    sdf_3d.surface_points()
    sdf_3d.scatter()
    plt.show()
    
def test_2d():
    sdf_2d_file_name = 'data/test/sdf/medium_black_spring_clamp_optimized_poisson_texture_mapped_mesh_clean_0.csv'
#    sdf_2d_file_name = 'data/test/sdf/brine_mini_soccer_ball_optimized_poisson_texture_mapped_mesh_clean_0.csv'
    sf2 = SdfFile(sdf_2d_file_name)
    sdf_2d = sf2.read()

    logging.info('Loaded 2D SDF %s'  %(sdf_2d_file_name))
    logging.info('Dimension: x: %d, y: %d' % sdf_2d.dimensions)
    logging.info('Origin: (%f, %f)' %(sdf_2d.origin[0], sdf_2d.origin[1]))
    logging.info('Resolution: %f' % sdf_2d.resolution)

    assert sdf_2d.dimensions[0] == 50
    assert sdf_2d.dimensions[1] == 50

    plt.figure()
    sdf_2d.vis_surface()
    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_2d()
    test_3d()
    

