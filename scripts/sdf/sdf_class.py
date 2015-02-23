""" 
Definition of SDF Class
Author: Sahaana Suri

**Currently assumes clean input**
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import IPython

from random_functions import has_p_and_n

from sys import version_info
if version_info[0] != 3:
    range = xrange

run_test_function = False

class SDF:
    def __init__(self, file_name):
        """
        Initializes the SDF class with data provided by a .sdf file
        Params: 
            file_name: string
        """
        my_file = open(file_name, 'r')
        self.file_name_ = file_name
        self.nx_, self.ny_, self.nz_ = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
        self.ox_, self.oy_, self.oz_ = [float(i) for i in my_file.readline().split()]   #shape origin
        self.dimension_ = self.nx_ * self.ny_ * self.nz_
        self.spacing_ = float(my_file.readline())
        self.data_ = np.zeros((self.nx_,self.ny_,self.nz_))
        self.xlist_, self.ylist_, self.zlist_, self.values_in_order_ = np.zeros(self.dimension_),\
                                                                            np.zeros(self.dimension_),\
                                                                                np.zeros(self.dimension_),\
                                                                                    np.zeros(self.dimension_)
        count = 0
        for k in range(self.nz_):
            for j in range(self.ny_):
                for i in range(self.nx_):
                    self.xlist_[count] = i + self.ox_
                    self.ylist_[count] = j + self.oy_
                    self.zlist_[count] = k + self.oz_
                    self.data_[i][j][k] = float(my_file.readline())
                    self.values_in_order_[count] = self.data_[i][j][k] 
                    count +=1 
        my_file.close()

    def file_name(self):
        """ 
        SDF file name information
        Params: -
        Returns: 
            (string): File name + path (relative to root from where loaded)
        """

        return self.file_name_

    def dimensions(self):
        """ 
        SDF dimension information
        Params: -
        Returns: 
            (int,int,int,int): the overall vector dimension (as for LSH), followed by the dimensions of the sdf
        """
        return self.dimension_,self.nx_, self.ny_, self.nz_

    def origin(self):
        """
        Object origin
        Params: - 
        Returns:
            (float, float, float): the object's origin
        """
        return self.ox_, self.oy_, self.oz_

    def spacing(self):
        """
        Object spacing
        Params: - 
        Returns:
            float: object spacing
        """
        return self.spacing_

    def data(self):
        """
        SDF value matrix. You can index into it, but will have to manually handle origin information to plot/etc. 
        Params: - 
        Returns:
            numpy.ndarray: nx x ny x nz matrix containing the SDF value of the object
        """
        return self.data_

    def make_windows(self, W, S, target=False): 
        """ 
        Function for windowing the SDF grid
        Params: 
            W: the side length of the window (currently assumed to be odd so centering makes sense)
            S: stride length between cubes (x axis "wraps" around)
            target: True for targetted windowing (filters for cubes containing both positive and negative values)
        Returns: 
            ([np.array,...,np.array]): contains a list of numpy arrays, each of which is an unrolled window/cube.
                                       window order based on center coordinate (increasing order of x, y, then z)
        """
        windows = []
        window_center = (W-1)/2 #assuming odd window
        offset = 0 #parameter used to "wrap around" x / not start back at x=1 each time  
        newx = 2*window_center + self.nx_
        newy = 2*window_center + self.ny_
        newz = 2*window_center + self.ny_
        padded_vals = np.zeros(newx*newy*newz)
     
        for k in range(self.nz_):
            for j in range(self.ny_):
                for i in range(self.nx_):
                    padded_vals[(i+window_center) + (j+window_center)*newx + (k+window_center)*newx*newy] = \
                                                            self.values_in_order_[i + j*self.nx_ + k*self.nx_*self.ny_]
        for z in range(window_center, self.nz_ + window_center, S):
            for y in range(window_center, self.ny_ + window_center, S):
                for x in range(window_center+offset, self.nx_ + window_center, S):
                        #print [map(lambda x: x-window_center,[x,y,z])]
                    new_window = np.empty(W**3)
                    count = 0
                    for k in range(-window_center, window_center+1):    
                        for j in range(-window_center, window_center+1):
                            for i in range(-window_center, window_center+1):
                                new_window[count] = padded_vals[(x+i) + (y+j)*newx + (z+k)*newx*newy] 
                                count += 1
                        windows.append(new_window)
                offset = (x+S) - (self.nx_+window_center)
        if target:
            windows = filter(has_p_and_n, windows)          
        return windows       

    def add_to_nearpy_engine(self, engine):
        """
        Inserts the SDF into the provided nearpy Engine
        Params: 
            engine: nearpy.engine.Engine 
        Returns: - 
        """
        engine.store_vector(self.values_in_order_,self.file_name_)
        #        print "Stored %s as vector" % self.file_name_

    def query_nearpy_engine(self, engine):
        """
        Queries the provided nearpy Engine for the SDF closest to this one
        Params:
            engine: nearpy.engine.Engine 
        Returns:
            (list (strings), list (tuples))
            list (strings): Names of the files that most closely match this one
            list (tuple): Additional information regarding the closest matches in (numpy.ndarray, string, numpy.float64) form:
                numpy.ndarray: the full vector of values of that match (equivalent to that SDF's "values_in_order_")
                string: the match's SDF's file name
                numpy.float64: the match's distance from this SDF
        """
        results = engine.neighbours(self.values_in_order_)
        file_names = [i[1] for i in results]
        return file_names, results

    def send_to_matlab(self, out_file):
        """
        Saves the SDF's coordinate and value information to the provided matlab workspace
        Params:
            out_file: string
        Returns: - 
        """
        scipy.io.savemat(out_file, mdict={'X':self.xlist_, 'Y': self.ylist_, 'Z': self.zlist_, 'vals': self.values_in_order_})
        "SDF information saved to %s" % out_file

    def make_plot(self):
        """
        Plots the SDF as a matplotlib 3D scatter plot, and displays the figure
        Params: - 
        Returns: - 
        """
        X, Y, Z, values = [],[],[], []
        for i in range(0,self.dimension_,1):
            val = self.values_in_order_[i] 
            if val <= 0: #Filter currently set to only show points on or within the surface of the object
                values.append(val)
                X.append(self.xlist_[i])
                Y.append(self.ylist_[i])
                Z.append(self.zlist_[i])
        fig = plt.figure()  
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter3D(X,Y,Z,c=values, cmap="Blues")
        plt.show()

 
def test_function():
    test_sdf = "aunt_jemima_original_syrup/processed/textured_meshes/optimized_tsdf_texture_mapped_mesh.sdf"
    matlab_file = "data.mat"
    teatime = SDF(test_sdf)
    print "Done processing %s" % test_sdf
    print "Dimension: %d, x: %d, y: %d, z: %d" % teatime.dimensions()
    print "Origin: (%f, %f, %f)" % teatime.origin()
    print "Spacing: %f" % teatime.spacing()
    print "Data Matrix: \n", teatime.data()

    #Testing LSH by just inserting the same sdf file twice, then querying it
    dimension = teatime.dimensions()[0]
    rbp = RandomBinaryProjections('rbp',10)
    engine = Engine(dimension, lshashes=[rbp])
    teatime.add_to_nearpy_engine(engine)
    teatime.add_to_nearpy_engine(engine)
    print "Query results: \n", teatime.query_nearpy_engine(engine)

    teatime.send_to_matlab(matlab_file)
    teatime.make_plot()

if run_test_function:
    test_function()

