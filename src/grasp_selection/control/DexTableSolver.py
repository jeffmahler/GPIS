from numpy import sin, cos, dot, pi, linspace
from numpy.linalg import norm
import matplotlib.pylab as plt
class DexTableSolver:

    DEFAULT_SAMPLE_SIZE = 1000

    @staticmethod
    def solve(r, d, phi, n = DEFAULT_SAMPLE_SIZE):
        return DexTableSolver.argmax(DexTableSolver.getInnerProductNormed(r, d, phi), 0, 2*pi, n)
            
    @staticmethod
    def getInnerProductNormed(r, d, phi):
        def innerProductNormed(theta):
            a = [r*cos(theta) - d, r*sin(theta)]
            g = [cos(theta + phi), sin(theta + phi)]
            return dot(a, g) / norm(a)
        return innerProductNormed
            
    @staticmethod
    def argmax(f, a, b, n):
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
    def _plot(f, a, b, n):
        x = linspace(a, b, n)
        plt.plot(x, [f(i) for i in x])
        plt.axis("tight")
        plt.show()