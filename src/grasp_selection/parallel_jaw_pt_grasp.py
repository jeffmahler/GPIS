import grasp

class ParallelJawPointGrasp(grasp.Grasp):
    def __init__(self, endpoints):
        # TODO: widen grasp from endpoints
        self.endpoints_ = endpoints

    def find_contacts(self, obj):
        # TODO: step along line of action and check intersection with object surface
        self.contacts_ = np.array([0,0])
        # TODO: return True if grasp contacts surface, False otherwise
        return True
