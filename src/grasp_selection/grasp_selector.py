from abc import ABCMeta, abstractmethod

class GraspSelector:
    __metaclass__ = ABCMeta

    def __init__(self, reward):
        self.reward_ = reward

    def best_grasp(self):
        '''
        Solves for the top grasp
        '''
        return self.top_K_grasps(1)

    @abstractmethod
    def top_K_grasps(self, K):
        '''
        Solves for the top grasp
        '''
        pass
