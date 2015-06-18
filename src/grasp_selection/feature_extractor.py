from abc import ABCMeta, abstractmethod

import feature_file as ff

class FeatureExtractor:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def extract(self, graspable):
        """ Returns a set of extracted features for a graspable """
        pass

class SHOTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, graspable):
        """ Returns a set of extracted SHOT features for a graspable """
        # TODO: make an OS call to extract features, then load using a feature file
        pass
