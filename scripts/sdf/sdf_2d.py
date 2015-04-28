class SDF2D:
	"""
	A representation of the signed distance function of a 2D object
	Author: Nikhil Sharma

	Adapted from 3D SDF class by Sahaana Suri
	"""

	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.io

	from nearpy import Engine
	import IPython

	from sys import version_info
	if version_info[0] != 3:
	    range = xrange

	class SDF:
	    def __init__(self, file_name):
	        """
	        Initializes the SDF class with data provided by a .sdf file
	        Params: 
	            file_name: string
	        """
	        my_file = open(file_name, 'r')
	        self.file_name_ = file_name
	        self.nx_, self.ny_ = [int(i) for i in my_file.readline().split()]     #dimension of each axis should all be equal for LSH
	        self.ox_, self.oy_ = [float(i) for i in my_file.readline().split()]   #shape origin
	        self.dimension_ = self.nx_ * self.ny_
	        self.spacing_ = float(my_file.readline())
	        self.data_ = np.zeros((self.nx_,self.ny_))
	        self.xlist_, self.ylist_, self.values_in_order_ = np.zeros(self.dimension_),\
	                                                                      		 np.zeros(self.dimension_),\
	                                                                                    np.zeros(self.dimension_)
	        count = 0
	        for k in range(self.nz_):
	            for j in range(self.ny_):
                    self.xlist_[count] = i + self.ox_
                    self.ylist_[count] = j + self.oy_
                    self.data_[i][j] = float(my_file.readline())
                    self.values_in_order_[count] = self.data_[i][j] 
                    count +=1 
	        my_file.close()
	        self.feature_vector_ = None #Kmeans feature representation


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
	        return self.dimension_,self.nx_, self.ny_

	    def origin(self):
	        """
	        Object origin
	        Params: - 
	        Returns:
	            (float, float, float): the object's origin
	        """
	        return self.ox_, self.oy_

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

	    def set_feature_vector(self, vector):
	        self.feature_vector = vector

	    def feature_vector(self):
	        return self.feature_vector      

	    def add_to_nearpy_engine(self, engine):
	        """
	        Inserts the SDF into the provided nearpy Engine
	        Params: 
	            engine: nearpy.engine.Engine 
	        Returns: - 
	        """
	        if self.feature_vector is None:
	            to_add = self.values_in_order
	        else:
	            to_add = self.feature_vector
	        #print to_add, type(to_add)
	        engine.store_vector(to_add,self.file_name_)
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
	        if self.feature_vector is None:
	            to_query = self.values_in_order
	        else:
	            to_query = self.feature_vector
	        results = engine.neighbours(to_query)
	        file_names = [i[1] for i in results]
	        return file_names, results

	    def send_to_matlab(self, out_file):
	        """
	        Saves the SDF's coordinate and value information to the provided matlab workspace
	        Params:
	            out_file: string
	        Returns: - 
	        """
	        scipy.io.savemat(out_file, mdict={'X':self.xlist_, 'Y': self.ylist_, 'vals': self.values_in_order_})
	        "SDF information saved to %s" % out_file

	    def make_plot(self):
	        """
	        Plots the SDF as a matplotlib 3D scatter plot, and displays the figure
	        Params: - 
	        Returns: - 
	        """
	        X, Y, values = [],[], []
	        for i in range(0,self.dimension_,1):
	            val = self.values_in_order_[i] 
	            if val <= 0: #Filter currently set to only show points on or within the surface of the object
	                values.append(val)
	                X.append(self.xlist_[i])
	                Y.append(self.ylist_[i])
	        fig = plt.figure()  
	        ax = fig.add_subplot(111,projection='3d')
	        ax.scatter3D(X,Y,c=values, cmap="Blues")
	        ax.set_xlim3d([0,self.nx_])
	        ax.set_ylim3d([0,self.ny_])
	        plt.show()

			




			





