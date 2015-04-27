'''
Reads and writes sdfs to file
Author: Jeff Mahler
'''
import numpy as np

class SdfFile:
    def __init__(self, file_name):
        self.file_name_ = file_name

    def read(self):
        '''
        Reads an SDF from file
        '''
        # read in basic params from file
        my_file = open(self.file_name_, 'r')
        nx, ny, nz = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
        ox, oy, oz = [float(i) for i in my_file.readline().split()]   #shape origin
        spacing = float(my_file.readline())
        sdf_data = np.zeros([nx, ny, nz])

        # loop through file, getting each value
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    sdf_data[i][j][k] = float(my_file.readline())
                    count += 1 
        my_file.close()

        return SDF(sdf_data, nx, ny, ox, oy, oz, spacing)

    def write(self, sdf):
        '''
        Writes an SDF to file
        '''
        todo = 1
