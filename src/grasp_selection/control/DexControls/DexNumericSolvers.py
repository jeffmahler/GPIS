from numpy import pi, arctan, sign
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
        
    @staticmethod
    def interpolate(n, origin, dest, speeds_ids, speeds, time_delta):        
        abs_deltas = [speeds[speeds_ids[i]] * time_delta for i in range(n)]        
        dirs = [sign(dest[i] - origin[i]) for i in range(n)]

        reached = [False] * n
        vectors = []
        current = origin[::]
        
        and_function = lambda x, y : x and y
        while not reduce(and_function, reached):       
            for i in range(n):
                if not reached[i]:
                    abs_delta = min(abs_deltas[i], abs(dest[i] - current[i]))
                    current[i] += abs_delta * dirs[i]
        
            vectors.append(current[::])
            
            for i in range(n):
                reached[i] = reached[i] or dirs[i] * (dest[i] - current[i]) <= 0

        return vectors