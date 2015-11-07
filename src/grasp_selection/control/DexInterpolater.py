from numpy import sign
class DexInterpolater:
    #Helper interpolator class

    @staticmethod
    def interpolate(origin, dest, speeds_ids, speeds, time_delta):
        n = len(origin)
        
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