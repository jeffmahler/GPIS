from numpy import pi, arctan
class DexNumericSolvers:

    @staticmethod
    def get_cartesian_angle(x, y):
        theta = 0
        if x == 0:
            if y >= 0:
                theta = pi / 2
            else: 
                theta = - pi / 2
        else:
            theta_ref = abs(arctan(y/x))
            if theta_ref > pi/2:
                theta_ref = pi - theta_ref

            if x >= 0 and y >= 0:
                theta = theta_ref
            elif y >= 0 and x < 0:
                theta = pi - theta_ref
            elif y < 0 and x < 0:
                theta = pi + theta_ref
            else:
                theta = 2*pi - theta_ref
                
        return theta