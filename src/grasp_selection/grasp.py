from abc import ABCMeta, abstractmethod

class Grasp:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.qualities = {}        
        self.pose = None

    @abstractmethod
    def find_contacts(self, obj):
        '''
        Finds the contact points on an object
        '''
        pass

    @abstractmethod
    def to_json(self):
        '''
        Converts a grasp to json
        '''
        pass

    @classmethod
    def from_json(cls, d):
        todo = 'Make this more general'

        pose = Pose()
        pose.position = Point(**d['gripper_pose']['position'])
        pose.orientation = Quaternion(**d['gripper_pose']['orientation'])

        return cls(pose, d['gripper_width'], d['flag'])

