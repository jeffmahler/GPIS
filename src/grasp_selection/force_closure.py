import reward

class ForceClosure(reward.Reward):
    def __init__(self):
        todo = 1

    def evaluate(self, object, grasp):
        fc = 1
        todo = 1
        return fc
