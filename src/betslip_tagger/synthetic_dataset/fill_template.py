import numpy as np


class RandomValueFinder:
    def __init__(self, values):
        self.values = values

    def get_random_value(self):
        num = np.random.randint(len(self.values))

        return self.values[num]
