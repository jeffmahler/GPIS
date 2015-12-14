from numpy import sin, cos, dot, pi, linspace, array
from DexNumericSolvers import DexNumericSolvers
from numpy.linalg import norm
from random import random
import matplotlib.pylab as plt
class DexTurntableSolver:

    DEFAULT_SAMPLE_SIZE = 1000

    @staticmethod
    def solve(r, d, phi, n = DEFAULT_SAMPLE_SIZE):
        return DexTurntableSolver._argmax(DexTurntableSolver._getInnerProductNormed(r, d, phi), 0, 2*pi, n)
            
    @staticmethod
    def _getInnerProductNormed(r, d, phi):
        def innerProductNormed(theta):
            a = [r*cos(theta) - d, r*sin(theta)]
            g = [cos(theta + phi), sin(theta + phi)]
            return dot(a, g) / norm(a)
        return innerProductNormed
            
    @staticmethod
    def _argmax(f, a, b, n):
        #finds the argmax x of f(x) in the range [a, b) with n samples
        delta = (b - a) / n
        max_y = f(a)
        max_x = a
        for i in range(1, n):
            x = i * delta
            y = f(x)
            if y >= max_y:
                max_y = y
                max_x = x
                
        return max_x
        
    @staticmethod
    def _plot_function(f, a, b, n):
        x = linspace(a, b, n)
        plt.plot(x, [f(i) for i in x])
        plt.axis("tight")
        plt.show()
        
def test_plots(n):
    def get_rand_in_range(a, b):
        return (b - a) * random() + a
    
    fig = plt.figure()
    
    for i in range(n):
        plt.clf()
        axis = plt.gca()
        axis.set_xlim([-20,20])
        axis.set_ylim([-20,20])
        
        x_o = get_rand_in_range(5, 15)
        y_o = get_rand_in_range(5, 15)
        theta_o =DexNumericSolvers.get_cartesian_angle(x_o, y_o)
        r = norm([x_o, y_o])
        phi = get_rand_in_range(0, 2 * pi)
                
        theta = DexTurntableSolver.solve(r, 20, phi)
        x = r * cos(theta)
        y = r * sin(theta)
                
        #vector to original obj pos
        v_obj_o = array([0, 0, x_o, y_o])
        
        #original vector of object in direction of grasp
        v_grasp_o = array([x_o, y_o, 5 * cos(phi + theta_o), 5 * sin(phi + theta_o)])
        
        #vector to obj pos
        v_obj = array([0, 0, x, y])
            
        #vector of object in direction of grasp
        v_grasp = array([x, y, 5 * cos(phi + theta), 5 * sin(phi + theta)])
        
        #vector of of arm to position of object
        v_arm = array([20, 0, x - 20, y])
               
        soa = array([v_arm, v_obj, v_grasp, v_obj_o, v_grasp_o]) 
        X,Y,U,V = zip(*soa)
        axis.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)

        plt.draw()
        fig.savefig('table_angles/%d.png' % i)

    

